"use strict";

const DATA = JSON.parse(document.getElementById("zoo-data").textContent);

const BY = new Map(DATA.objects.map(o => [ o.object_id, o ]));

const SOURCES = new Map(DATA.sources.map(s => [ s.id, s ]));

const $ = s => document.querySelector(s);

const esc = v => String(v ?? "").replace(/[&<>"']/g, c => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
}[c]));

const txt = v => Array.isArray(v) ? v.map(txt).join(" / ") : v && typeof v === "object" ? Object.entries(v).map(([k, x]) => k + ": " + txt(x)).join(" \xb7 ") : v == null ? "\u2014" : typeof v === "number" ? String(Number(v.toPrecision(6))) : String(v);

const DOMAIN = {
    household: "Household",
    industry: "Industry",
    laboratory: "Laboratory"
};

const ORIGIN = {
    reported: "Reported",
    derived: "Derived",
    calibrated: "Calibrated",
    transferred: "Transferred",
    assumed: "Assumed",
    unresolved: "Unresolved",
    proxy: "Proxy"
};

const ROUTE = {
    reported_material: "Reported material",
    analogous_transfer: "Analogous transfer",
    test_calibration: "Test calibration",
    constitutive_geometry: "Constitutive law + geometry"
};

let active = null, refNumbers = new Map;

const REDUNDANT_CONTEXT = new Set([ "Material constants adopted as reported in the source." ]);

const link = (url, label) => `<a href="${esc(url)}" target="_blank" rel="noopener">${esc(label)}</a>`;

function refs(ids = []) {
    return `<span class="source-links">${ids.map(id => `<a href="#ref-${esc(id)}" data-ref="${esc(id)}">[${refNumbers.get(id)}]</a>`).join("")}</span>`;
}

function facts(rows) {
    return `<dl class="facts">${rows.filter(([, v]) => v != null && txt(v) !== "" && (!Array.isArray(v) || v.length)).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(txt(v))}</dd>`).join("")}</dl>`;
}

function table(heads, rows) {
    return `<div class="table-scroll"><table><thead><tr>${heads.map(x => `<th>${esc(x)}</th>`).join("")}</tr></thead><tbody>${rows.map(row => `<tr>${row.map(x => `<td>${x}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}

function properties(rows) {
    return table([ "Property / region", "Value / unit", "Origin", "Source" ], rows.map(p => [ `<div class="parameter">${esc(p.parameter)}${p.region ? `<div>${esc(p.region)}</div>` : ""}${p.context && !REDUNDANT_CONTEXT.has(p.context) ? `<div class="conditions">${esc(p.context)}</div>` : ""}</div>`, `<span class="value">${esc(txt(p.value))}${p.unit ? " " + esc(p.unit) : ""}</span>`, esc(ORIGIN[p.origin] || p.origin), refs(p.source_ids) ]));
}

function source(s) {
    return `<article class="ref" id="ref-${esc(s.id)}"><div>[${refNumbers.get(s.id)}] ${s.url ? link(s.url, s.title || s.id) : esc(s.title || s.id)}</div><div class="ref-meta">${esc(txt(s.authors || s.organization || ""))}${s.year ? " \xb7 " + esc(s.year) : ""}${s.doi ? " \xb7 DOI: " + esc(s.doi) : ""}${s.arxiv ? " \xb7 arXiv: " + esc(s.arxiv) : ""}</div></article>`;
}

function compare(c) {
    return `<div class="record"><div class="record-title">${esc(c.scope)}</div>${facts([ [ "Role", c.role ], [ "Compared with", c.comparison_target ] ])}${c.response_agreement === false ? `<p>${"Response mismatch \xb7 excluded from L1 support"}</p>` : ""}${c.results.length ? table([ "Response", "Reference", "Computed", "Difference" ], c.results.map(r => [ esc(r.quantity), esc(txt(r.reference)) + (r.reference != null ? " " + esc(r.unit) : ""), esc(txt(r.computed)) + (r.computed != null ? " " + esc(r.unit) : ""), esc(txt(r.difference)) ])) : ""}${facts([ [ "Scope", c.limitation ] ])}${refs(c.source_ids)}${c.record ? link(c.record, "Result") : ""}</div>`;
}

function showObject(id) {
    const o = BY.get(id);
    if (!o) throw Error("Unknown Zoo object: " + id);
    active = o;
    window.disposeZooMesh?.();
    const s = o.supporting_references, m = o.mesh, g = m.geometry, ps = o.material_properties;
    refNumbers = new Map(s.sources.map((id, i) => [ id, i + 1 ]));
    $("#detail-title").textContent = o.display_name;
    $("#detail-body").innerHTML = `<aside class="visual-column"><img class="detail-thumbnail" src="${esc(m.thumbnail)}" alt="${esc(o.display_name)}"><div id="mesh-stage" class="mesh-stage" hidden></div><div class="visual-controls"><button data-view-mesh="${esc(id)}">${"View 3D"}</button>${m.physical_geometry ? `<button data-view-mesh="${esc(id)}" data-structure="1">${"Structure"}</button>` : ""}<a href="${esc(m.visual.url)}" download>${"Mesh"}</a></div><p id="mesh-status" class="mesh-status" aria-live="polite"></p></aside><div class="content-column">\n <section class="object-section"><h3>${"Name"}</h3><h2>${esc(o.display_name)}</h2>${facts([ [ "Category", DOMAIN[o.name.domain] ] ])}</section>\n <section class="object-section"><h3>${"Mesh & geometry"}</h3>${facts([ [ "Mesh bounds", m.extents_mm.length ? m.extents_mm.map(x => Number(x).toFixed(1)).join(" \xd7 ") + " mm (X \xd7 Y \xd7 Z)" : null ], [ "Structure", g.structure ], [ "Wall / layer thickness", g.wall_thickness_mm != null ? txt(g.wall_thickness_mm) + " mm" : g.thickness_record ], [ "Dimensions (mm)", g.dimensions_mm ], [ "Ranges (mm)", g.dimension_ranges_mm ] ])}${refs(g.source_ids)}</section>\n <section class="object-section"><h3>${"Material properties"}</h3>${facts([ [ "Model", ps.law ], [ "Scope", ps.scope ], [ "Input readiness", ps.input_readiness ] ])}${ps.parameters.some(p => p.stage === "catalog_scenario") ? `<h4>${"Selected properties and data"}</h4>${properties(ps.parameters.filter(p => p.stage === "catalog_scenario"))}` : ""}${ps.linked_parameters?.length ? `<h4>${"Linked surrogate material law"}</h4>${properties(ps.linked_parameters)}` : ""}${ps.parameters.filter(p => p.stage === "registered_record").length ? properties(ps.parameters.filter(p => p.stage === "registered_record")) : ps.parameters.some(p => p.stage === "catalog_scenario") ? "" : `<p class="empty">${"No registered material parameters."}</p>`}\n ${ps.parameters.some(p => p.stage === "validation_scenario") ? `<h4>${"Validation scenario inputs"}</h4>${properties(ps.parameters.filter(p => p.stage === "validation_scenario"))}` : ""}\n ${ps.reported_response_inputs?.length ? `<details><summary>${"Reported values before calibration or transfer"}</summary>${properties(ps.reported_response_inputs)}</details>` : ""}\n \n ${ps.parameters.some(p => p.stage === "literature_candidate") ? `<details><summary>${"Literature candidates \xb7 not adopted"}</summary>${properties(ps.parameters.filter(p => p.stage === "literature_candidate"))}</details>` : ""}</section>\n <section class="object-section"><h3>${"Supporting references"}</h3>${s.scope_review?.whole_object_transfer_level ? facts([ [ "Selected response-scope level", s.scope_review.level ], [ "Selected scope", s.scope_review.scope ], [ "Object/product transfer level", s.scope_review.whole_object_transfer_level ] ]) : ""}${s.evidence.map(ev => facts([ [ "Evidence level", ev.level ], [ "Response / region", ev.scope ], [ "Route", ev.routes.map(x => ROUTE[x] || x) ], [ "Status", ev.status ] ])).join("")}\n ${s.comparisons.length ? `<h4>${"Comparisons"}</h4>${s.comparisons.map(compare).join("")}` : ""}\n ${s.calibrations.length ? `<h4>${"Calibration"}</h4>${s.calibrations.map(c => `<div class="record">${facts([ [ "Performed by", c.performed_by ], [ "Adjusted parameters", c.adjusted ], [ "Fixed inputs", c.fixed ], [ "Target", c.target ], [ "Output", c.output ], [ "Method", c.method ] ])}${c.calibrated_on ? refs([ c.calibrated_on ]) : ""}</div>`).join("")}` : ""}\n ${s.simulations.length ? `<h4>${"Simulations"}</h4>${s.simulations.map(r => `<div class="record">${facts([ [ "Solver", r.solver ], [ "Response / region", r.scope ], [ "Status", r.status ] ])}${[r.protocol ? link(r.protocol, "Protocol") : "", r.record ? link(r.record, "Result") : ""].filter(Boolean).join(" \xb7 ")}</div>`).join("")}` : ""}\n ${s.tests.length ? `<details><summary>${"Published tests"}</summary>${s.tests.map(t => `<div class="record"><div class="record-title">${esc(t.quantity)}</div>${facts([ [ "Response", txt(t.value) + " " + (t.unit || "") ], [ "Conditions", t.test ], [ "Event", t.event ], [ "Scope", t.limits || t.limitation ] ])}${refs(t.source_ids)}</div>`).join("")}</details>` : ""}\n ${s.proxy_links.length ? `<details><summary>${"Material transfers"}</summary>${s.proxy_links.map(p => `<div class="record">${facts([ [ "Transferred from", p.object ], [ "Transferred property", p.transfer ], [ "Basis", p.justification ], [ "Scope", p.limits ] ])}${refs(p.source_ids)}</div>`).join("")}</details>` : ""}\n <details><summary>${"Assessment criteria"}</summary>${s.assessment_criteria.map(c => facts([ [ "Origin", c.origin ], [ "Criterion", c.criterion ] ])).join("")}</details>\n ${s.not_supported?.length ? `<details><summary>${"Not supported"}</summary><ul>${s.not_supported.map(x => `<li>${esc(x)}</li>`).join("")}</ul></details>` : ""}<h4>${"Sources"}</h4>${s.sources.map(id => source(SOURCES.get(id))).join("")}</section></div>`;
    if (!$("#detail").open) $("#detail").showModal();
    $("#detail").scrollTop = 0;
}

$("#gallery").innerHTML = Object.entries(DOMAIN).map(([domain, label]) => `<section class="category" id="${domain}"><h2>${label}</h2><div class="grid">${DATA.objects.filter(o => o.name.domain === domain).map(o => `<button class="object-tile" data-object="${esc(o.object_id)}" aria-label="${esc(o.display_name)}"><img src="${esc(o.mesh.thumbnail)}" alt="${esc(o.display_name)}" loading="lazy"><span>${esc(o.display_name)}</span></button>`).join("")}</div></section>`).join("");

document.addEventListener("click", e => {
    const b = e.target.closest("[data-object]");
    if (b) {
        showObject(b.dataset.object);
        history.replaceState(null, "", "#object=" + encodeURIComponent(b.dataset.object));
    }
    const ref = e.target.closest("[data-ref]");
    if (ref) {
        e.preventDefault();
        document.getElementById("ref-" + ref.dataset.ref).scrollIntoView({
            block: "start"
        });
    }
});

$("#close-detail").onclick = () => $("#detail").close();

$("#detail").addEventListener("close", () => {
    if (!$("#detail").open) {
        window.disposeZooMesh?.();
        history.replaceState(null, "", location.pathname + location.search);
    }
});

function linked() {
    if (location.hash.startsWith("#object=")) showObject(decodeURIComponent(location.hash.slice(8)));
}

window.addEventListener("hashchange", linked);

linked();
