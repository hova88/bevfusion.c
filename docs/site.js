const stages={input:{shape:"[6, 3, 256, 704] + [P, 5]",name:"Normalized cameras + timed points",copy:"Six NCHW cameras, a dynamic point table, and 210 calibration floats enter from the CRC-protected BFI mapping.",memory:"host mmap → one input-boundary H2D"},camera:{shape:"[1, 80, 180, 180]",name:"Metric image BEV",copy:"Swin-T builds three scales. FPN, categorical depth, calibration-aware lift-splat, and downsampling place context on the ground grid.",memory:"GPU resident · no intermediate host transfer"},lidar:{shape:"[1, 256, 180, 180]",name:"Sparse-to-dense LiDAR BEV",copy:"Stable voxel grouping, MeanVFE, 21 sparse convolutions, and height compression preserve deterministic order before dense BEV.",memory:"GPU resident · bounded voxels and sparse sites"},fusion:{shape:"[1, 336, 180, 180]",name:"Channel-concatenated sensor features",copy:"The branches already share metric cells. Concatenation needs no learned geometric alignment; the BEV backbone creates shared features and a heatmap.",memory:"GPU resident · camera 80ch + LiDAR 256ch"},output:{shape:"≤ 200 × 11 fields",name:"Canonical metric detections",copy:"TransFusion decodes center, dimensions, yaw, velocity, score, and one of ten classes. Rotated filtering produces the public C structure and JSON.",memory:"8,804-byte D2H boundary → JSON / TUI"}};
const tabs=document.querySelectorAll("[data-stage]");for(const tab of tabs){tab.addEventListener("click",()=>{for(const other of tabs)other.setAttribute("aria-selected",String(other===tab));const s=stages[tab.dataset.stage];document.querySelector("#tensor-shape").textContent=s.shape;document.querySelector("#tensor-name").textContent=s.name;document.querySelector("#tensor-copy").textContent=s.copy;document.querySelector("#tensor-memory").textContent=s.memory})}

const modelRoot=document.querySelector("#model-graph");
if(modelRoot){
  const number=new Intl.NumberFormat("en-US");
  const compact=new Intl.NumberFormat("en-US",{notation:"compact",maximumFractionDigits:2});
  const formatBytes=value=>`${(value/1048576).toFixed(2)} MiB`;
  const el=(tag,className,text)=>{const node=document.createElement(tag);if(className)node.className=className;if(text!==undefined)node.textContent=text;return node};
  let graph=null,activeStage="all",visibleLimit=24;
  const flowRoot=document.querySelector("#model-flow"),filters=document.querySelector("#stage-filters");
  const list=document.querySelector("#operator-list"),overview=document.querySelector("#stage-overview");
  const search=document.querySelector("#model-search"),empty=document.querySelector("#model-empty");

  function selectFlow(item,button){
    for(const node of flowRoot.children)node.setAttribute("aria-current",String(node===button));
    document.querySelector("#flow-name").textContent=item.operator;
    document.querySelector("#flow-shape").textContent=item.output;
    document.querySelector("#flow-note").textContent=`${item.input} → ${item.output} · ${item.note}`;
  }
  function renderFlow(){
    graph.flow.forEach((item,index)=>{
      const button=el("button","flow-step");button.type="button";button.setAttribute("aria-current",String(index===0));
      button.append(el("small","",String(index+1).padStart(2,"0")),el("strong","",item.operator),el("code","",item.output));
      button.addEventListener("click",()=>selectFlow(item,button));flowRoot.append(button);
    });
  }
  function selectStage(id){
    activeStage=id;visibleLimit=24;
    for(const button of filters.children)button.setAttribute("aria-pressed",String(button.dataset.stage===id));
    renderModules();
  }
  function renderFilters(){
    const all={id:"all",title:"All branches"};
    [all,...graph.stages].forEach(stage=>{const button=el("button","",stage.title);button.type="button";button.dataset.stage=stage.id;button.setAttribute("aria-pressed",String(stage.id==="all"));button.addEventListener("click",()=>selectStage(stage.id));filters.append(button)});
  }
  function renderModules(){
    const query=search.value.trim().toLowerCase();
    const matches=graph.modules.filter(module=>(activeStage==="all"||module.stage===activeStage)&&(!query||`${module.name} ${module.operator} ${module.tensor_shapes.flat().join(" ")}`.toLowerCase().includes(query)));
    const stage=graph.stages.find(item=>item.id===activeStage);
    const elements=matches.reduce((sum,item)=>sum+item.elements,0),bytes=matches.reduce((sum,item)=>sum+item.bytes,0);
    overview.replaceChildren();
    const title=el("div");title.append(el("span","explorer-kicker",stage?stage.title:"Entire checkpoint"),el("strong","",`${matches.length} entries · ${compact.format(elements)} elements · ${formatBytes(bytes)}`));
    overview.append(title);if(stage)overview.append(el("p","",stage.description));
    list.replaceChildren();empty.hidden=matches.length!==0;
    matches.slice(0,visibleLimit).forEach(module=>{
      const details=el("details","operator-row"),summary=el("summary");
      summary.append(el("span","operator-kind",module.operator),el("code","operator-name",module.name),el("code","operator-shape",module.tensor_shapes.map(shape=>`[${shape.join(", ")}]`).join(" · ")),el("span","operator-count",number.format(module.elements)));
      const body=el("div","operator-detail");body.append(el("p","","Stored state tensors"));
      const names=el("ul");module.tensor_names.forEach((name,index)=>{const item=el("li");item.append(el("code","",name),el("span","",`${module.dtypes.join(" / ")} · [${module.tensor_shapes[index].join(", ")}]`));names.append(item)});body.append(names);details.append(summary,body);list.append(details);
    });
    if(matches.length>visibleLimit){const more=el("button","model-more",`Show all ${matches.length} entries`);more.type="button";more.addEventListener("click",()=>{visibleLimit=matches.length;renderModules()});list.append(more)}
  }
  fetch("model-graph.json").then(response=>{if(!response.ok)throw new Error(`HTTP ${response.status}`);return response.json()}).then(data=>{
    graph=data;document.querySelector("#model-modules").textContent=number.format(data.summary.modules);document.querySelector("#model-tensors").textContent=number.format(data.summary.tensors);document.querySelector("#model-elements").textContent=compact.format(data.summary.elements);document.querySelector("#model-bytes").textContent=formatBytes(data.summary.bytes);
    renderFlow();renderFilters();renderModules();search.addEventListener("input",()=>{visibleLimit=24;renderModules()});
  }).catch(error=>{overview.textContent=`The generated model data could not be loaded (${error.message}). Open the Markdown inventory below.`});
}
