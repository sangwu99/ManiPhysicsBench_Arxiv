import { VisualScene } from "./visual_scene.js";

let scene = null;

window.disposeZooMesh = () => {
    scene?.dispose();
    scene = null;
};

document.addEventListener("click", async e => {
    const button = e.target.closest("[data-view-mesh]");
    if (!button) return;
    const object = BY.get(button.dataset.viewMesh), stage = document.getElementById("mesh-stage"), status = document.getElementById("mesh-status");
    window.disposeZooMesh();
    stage.hidden = false;
    document.querySelector(".detail-thumbnail").hidden = true;
    status.textContent = "Loading\u2026";
    const current = new VisualScene(stage);
    scene = current;
    current.scene.background.set("#ffffff");
    try {
        const loaded = await current.load(button.dataset.structure ? object.mesh.physical_geometry : object.mesh.visual);
        if (!loaded) return;
        status.textContent = "Drag to rotate \xb7 Scroll to zoom";
    } catch (error) {
        stage.dataset.error = error.message;
        status.textContent = "Unable to load mesh: " + error.message;
        console.error(error);
    }
});
