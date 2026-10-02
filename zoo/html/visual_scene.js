import * as THREE from "three";

import { OrbitControls } from "./vendor/OrbitControls.js";

import { GLTFLoader } from "./vendor/loaders/GLTFLoader.js";

import { ColladaLoader } from "./vendor/loaders/ColladaLoader.js";

import { RoomEnvironment } from "./vendor/environments/RoomEnvironment.js";

function disposeObject(root) {
    const geometries = new Set, materials = new Set, textures = new Set;
    root?.traverse(o => {
        if (o.geometry) geometries.add(o.geometry);
        for (const m of [ o.material ].flat().filter(Boolean)) materials.add(m);
    });
    for (const m of materials) {
        for (const v of Object.values(m)) if (v?.isTexture) textures.add(v);
        m.dispose();
    }
    for (const g of geometries) g.dispose();
    for (const t of textures) {
        t.dispose();
        t.source?.data?.close?.();
    }
}

async function loadVisual(spec) {
    const manager = new THREE.LoadingManager;
    let complete;
    const failures = [];
    const loaded = new Promise(resolve => complete = resolve);
    manager.onLoad = complete;
    manager.onError = url => failures.push(url);
    let object;
    if (spec.format === "glb") object = (await new GLTFLoader(manager).loadAsync(spec.url)).scene; else if (spec.format === "dae") object = (await new ColladaLoader(manager).loadAsync(spec.url)).scene; else throw new Error("Unsupported visual format: " + spec.format);
    await loaded;
    if (failures.length) {
        disposeObject(object);
        throw new Error("Missing visual dependencies: " + failures.join(", "));
    }
    if (spec.post_rotation_x) object.rotateX(spec.post_rotation_x);
    return object;
}

export class VisualScene {
    constructor(target, {interactive: interactive = true, size: size = null} = {}) {
        this.target = target;
        this.generation = 0;
        this.object = null;
        this.disposed = false;
        this.scene = new THREE.Scene;
        this.scene.background = new THREE.Color("#edf2ed");
        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            preserveDrawingBuffer: true
        });
        this.renderer.setPixelRatio(size ? 1 : Math.min(devicePixelRatio, 2));
        this.renderer.outputColorSpace = THREE.SRGBColorSpace;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1;
        target.replaceChildren(this.renderer.domElement);
        const room = new RoomEnvironment(this.renderer), pmrem = new THREE.PMREMGenerator(this.renderer);
        this.environment = pmrem.fromScene(room, .04);
        this.scene.environment = this.environment.texture;
        room.dispose();
        pmrem.dispose();
        this.scene.add(new THREE.HemisphereLight(16777215, 8096129, 1));
        const key = new THREE.DirectionalLight(16777215, 1.8);
        key.position.set(2, -3, 4);
        this.scene.add(key);
        this.camera = new THREE.PerspectiveCamera(34, 1, 1e-5, 100);
        this.camera.up.set(0, 0, 1);
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enabled = interactive;
        this.controls.addEventListener("change", () => this.render());
        const resize = () => {
            const w = size || target.clientWidth, h = size || target.clientHeight;
            this.camera.aspect = w / h;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(w, h);
            if (this.object) this.frameObject();
        };
        resize();
        if (!size) {
            this.observer = new ResizeObserver(resize);
            this.observer.observe(target);
        }
    }
    async load(spec) {
        const generation = ++this.generation;
        delete this.target.dataset.ready;
        delete this.target.dataset.error;
        const content = await loadVisual(spec);
        if (this.disposed || generation !== this.generation) {
            disposeObject(content);
            return null;
        }
        if (this.object) {
            this.scene.remove(this.object);
            disposeObject(this.object);
        }
        this.object = new THREE.Group;
        this.object.add(content);
        this.object.updateMatrixWorld(true);
        const bounds = (new THREE.Box3).setFromObject(this.object, true), center = bounds.getCenter(new THREE.Vector3);
        const size = bounds.getSize(new THREE.Vector3);
        if (!Number.isFinite(size.length()) || size.length() <= 0) {
            disposeObject(this.object);
            throw new Error("Empty visual bounds");
        }
        this.object.position.sub(center);
        this.scene.add(this.object);
        this.radius = size.length() / 2;
        const materials = new Set, textureMaps = new Set;
        let meshes = 0, triangles = 0;
        content.traverse(o => {
            if (!o.isMesh) return;
            meshes++;
            triangles += (o.geometry.index?.count || o.geometry.attributes.position.count) / 3;
            for (const m of [ o.material ].flat()) {
                materials.add(m);
                if (m.map) textureMaps.add(m.map);
            }
        });
        this.frameObject();
        this.render();
        this.target.dataset.ready = "true";
        this.info = {
            source: spec.source,
            signature: spec.signature,
            extents_mm: size.toArray().map(x => x * 1e3),
            meshes: meshes,
            triangles: triangles,
            materials: materials.size,
            base_color_textures: textureMaps.size,
            colors: [ ...materials ].map(m => m.color?.getHexString()).filter(Boolean)
        };
        return this.info;
    }
    frameObject() {
        const v = THREE.MathUtils.degToRad(this.camera.fov), h = 2 * Math.atan(Math.tan(v / 2) * this.camera.aspect);
        const distance = this.radius / Math.sin(Math.min(v, h) / 2) * 1.08;
        this.camera.position.copy(new THREE.Vector3(1.3, -1.7, 1.05).normalize().multiplyScalar(distance));
        this.camera.near = distance / 1e3;
        this.camera.far = distance * 100;
        this.camera.updateProjectionMatrix();
        this.controls.target.set(0, 0, 0);
        this.controls.minDistance = this.radius * 1.1;
        this.controls.maxDistance = this.radius * 15;
        this.controls.update();
        this.render();
    }
    render() {
        if (!this.disposed) this.renderer.render(this.scene, this.camera);
    }
    png() {
        this.render();
        return this.renderer.domElement.toDataURL("image/png");
    }
    dispose() {
        this.disposed = true;
        this.generation++;
        this.observer?.disconnect();
        this.controls.dispose();
        disposeObject(this.object);
        this.environment.dispose();
        this.renderer.dispose();
        this.renderer.forceContextLoss();
    }
}
