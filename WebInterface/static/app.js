/* Native browser UI with the locally installed Plotly bundle; no build step. */
'use strict';
const $ = (id) => document.getElementById(id);
const esc = (value) => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = (value) => value == null ? 'Not available' : Number(value).toLocaleString(undefined, {maximumSignificantDigits: 4});
const pretty = (value) => String(value).replaceAll('_', ' ').replaceAll('-', ' ');
let saved;
try { saved = JSON.parse(localStorage.getItem('metasymbo') || '{}'); } catch { saved = {}; }
const state = {page:'design', view:'structure', current:null, items:[], library:[], offset:0, total:0, capabilities:{}, runs:[],
  brief: typeof saved.brief === 'string' ? saved.brief : '',
  favorites: Array.isArray(saved.favorites) ? saved.favorites : [], compare: Array.isArray(saved.compare) ? saved.compare.slice(0,4) : [],
  selected: saved.selected || null, validation: saved.validation || {}, catalog: new Map()};
let toastTimer, searchTimer, libraryRequest = 0, runSignature = '', polling = false, submitting = false;

function persist() {
  try { localStorage.setItem('metasymbo', JSON.stringify({brief:state.brief, favorites:state.favorites, compare:state.compare, selected:state.current?.id, validation:state.validation})); }
  catch { toast('Browser storage is unavailable. Export your brief and candidate to retain them.', true); }
}
function toast(message, error=false) {
  clearTimeout(toastTimer); $('toast').textContent=message; $('toast').classList.toggle('error', error); $('toast').hidden=false;
  toastTimer=setTimeout(()=>{$('toast').hidden=true;},6000);
}
async function api(path, options={}) {
  const response=await fetch('/api'+path, options);
  if (!response.ok) {
    let message='The request could not complete.';
    try { const body=await response.json(); message=typeof body.detail==='string'?body.detail:'Check the submitted values and try again.'; } catch {}
    throw new Error(message);
  }
  return response.json();
}
function cache(items) { items.filter(x=>!x.error).forEach(x=>state.catalog.set(x.id,x)); }
async function getCandidate(id) {
  if (state.catalog.has(id)) return state.catalog.get(id);
  const item=await api('/candidates/'+encodeURIComponent(id)); cache([item]); return item;
}

function thumbnail(item) {
  if (!item?.cart_coords) return '<svg viewBox="0 0 180 110" aria-hidden="true"><text x="90" y="60" text-anchor="middle" fill="#aabbbf">Unavailable</text></svg>';
  const project=p=>[.8*p[0]-.6*p[1], -.3*p[0]-.35*p[1]+.85*p[2]];
  const points=item.cart_coords.map(project);
  const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);
  const low=[Math.min(...xs),Math.min(...ys)],high=[Math.max(...xs),Math.max(...ys)];
  const scale=Math.min(135/Math.max(high[0]-low[0],.001),78/Math.max(high[1]-low[1],.001));
  const pos=p=>[90+(p[0]-(low[0]+high[0])/2)*scale,55-(p[1]-(low[1]+high[1])/2)*scale];
  const p=points.map(pos);
  const lines=item.edges.map(([a,b])=>`<line x1="${p[a][0]}" y1="${p[a][1]}" x2="${p[b][0]}" y2="${p[b][1]}"/>`).join('');
  return `<svg viewBox="0 0 180 110" role="img" aria-label="${esc(item.nodes)} nodes and ${esc(item.struts)} struts"><g stroke="#24858b" stroke-width="2.5" stroke-linecap="round">${lines}</g><g fill="#176f78" stroke="#e9f4f3" stroke-width=".65">${p.map(([x,y])=>`<circle cx="${x}" cy="${y}" r="2.3"/>`).join('')}</g></svg>`;
}
const add=(a,b)=>a.map((v,i)=>v+b[i]);
const sub=(a,b)=>a.map((v,i)=>v-b[i]);
const mul=(a,s)=>a.map(v=>v*s);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const unit=a=>mul(a,1/(Math.hypot(...a)||1));
const cellPoint=(basis,x,y,z)=>add(add(mul(basis[0],x),mul(basis[1],y)),mul(basis[2],z));

function latticePlot(item, repeats=1, commonSpan=null) {
  const points=[], edges=[];
  for(let x=0;x<repeats;x++)for(let y=0;y<repeats;y++)for(let z=0;z<repeats;z++){
    const offset=cellPoint(item.basis,x,y,z), index=points.length;
    points.push(...item.cart_coords.map(p=>add(p,offset)));
    edges.push(...item.edges.map(([a,b])=>[a+index,b+index]));
  }
  const corners=[];
  for(let x=0;x<2;x++)for(let y=0;y<2;y++)for(let z=0;z<2;z++)corners.push(cellPoint(item.basis,x*repeats,y*repeats,z*repeats));
  const bounds=[0,1,2].map(i=>{const values=points.concat(corners).map(p=>p[i]);return [Math.min(...values),Math.max(...values)];});
  const span=commonSpan||Math.max(...bounds.map(([a,b])=>b-a),.1);
  const radius=Math.max(...item.lengths)*.009;
  const mesh={type:'mesh3d',x:[],y:[],z:[],i:[],j:[],k:[],color:'#32999d',hoverinfo:'skip',name:'Struts',showlegend:false,
    lighting:{ambient:.65,diffuse:.8,specular:.4,roughness:.55},lightposition:{x:100,y:150,z:300},flatshading:false};
  const sides=8;
  for(const [a,b] of edges) {
    const p=points[a],q=points[b],direction=unit(sub(q,p));
    if(Math.hypot(...sub(q,p))<1e-9)continue;
    const u=unit(cross(direction,Math.abs(direction[2])>.9?[0,1,0]:[0,0,1])),v=cross(direction,u),start=mesh.x.length;
    for(const center of [p,q])for(let n=0;n<sides;n++){
      const t=2*Math.PI*n/sides,vert=add(center,add(mul(u,radius*Math.cos(t)),mul(v,radius*Math.sin(t))));
      mesh.x.push(vert[0]);mesh.y.push(vert[1]);mesh.z.push(vert[2]);
    }
    for(let n=0;n<sides;n++){const next=(n+1)%sides;mesh.i.push(start+n,start+next);mesh.j.push(start+next,start+sides+next);mesh.k.push(start+sides+n,start+sides+n);}
  }
  const nodes={type:'scatter3d',mode:'markers',x:points.map(p=>p[0]),y:points.map(p=>p[1]),z:points.map(p=>p[2]),
    customdata:points.map((_,i)=>i%item.nodes),marker:{size:3.6,color:'#157d84',line:{color:'#91c2c2',width:.5}},name:'Nodes',showlegend:false,
    hovertemplate:'Node %{customdata}<br>x %{x:.4f}<br>y %{y:.4f}<br>z %{z:.4f}<extra>Cell coordinates</extra>'};
  const cell={type:'scatter3d',mode:'lines',x:[],y:[],z:[],line:{color:'#bdd0d4',width:2,dash:'dash'},hoverinfo:'skip',showlegend:false};
  for(let a=0;a<8;a++)for(const bit of [1,2,4])if(!(a&bit)){
    const b=a|bit;['x','y','z'].forEach((key,i)=>cell[key].push(corners[a][i],corners[b][i],null));
  }
  const origin=bounds.map(([a])=>a-span*.13),axisLength=span*.16;
  const axes=[['X','#b78574'],['Y','#76a092'],['Z','#808bbe']].map(([label,color],i)=>{
    const end=[...origin];end[i]+=axisLength;
    return {type:'scatter3d',mode:'lines+text',x:[origin[0],end[0]],y:[origin[1],end[1]],z:[origin[2],end[2]],text:['',label],textposition:'top center',textfont:{size:10,color},line:{color,width:3},hoverinfo:'skip',showlegend:false};
  });
  const scene={bgcolor:'white',aspectmode:'cube',camera:{eye:{x:.9,y:1.05,z:.7},up:{x:0,y:0,z:1},projection:{type:'perspective'}}};
  ['xaxis','yaxis','zaxis'].forEach((key,i)=>{const center=(bounds[i][0]+bounds[i][1])/2;scene[key]={visible:false,range:[center-span*.69,center+span*.69]};});
  return {data:[mesh,nodes,cell,...axes],layout:{margin:{l:0,r:0,t:0,b:0},paper_bgcolor:'white',scene,uirevision:'lattice-camera',showlegend:false}};
}
const plotConfig={responsive:true,displayModeBar:false,scrollZoom:true};
async function renderViewer() {
  if(!state.current)return;
  let repeats=Number($('repeat').value);
  if(state.current.struts*repeats**3>15000){repeats=1;$('repeat').value='1';toast('This lattice is large; showing one unit cell to keep the view responsive.');}
  const {data,layout}=latticePlot(state.current,repeats);
  await Plotly.react('lattice-view',data,layout,plotConfig);
  $('lattice-view').removeAllListeners('plotly_click');
  $('lattice-view').on('plotly_click',event=>{const point=event.points[0];if(point.data.name==='Nodes')selectNode(point.customdata);});
  $('view-caption').textContent=repeats===1?'Real project geometry · physical scale unverified':`${repeats} × ${repeats} × ${repeats} display only · periodic connectivity unverified`;
}
function selectNode(index) {
  const point=state.current.frac_coords[index];
  if(!point)return;
  $('node-selection').textContent=`Node ${index} · fractional (${point.map(fmt).join(', ')}) · ${state.current.degrees[index]} neighbors`;
  $('node-selector').value=String(index);
  const count=state.current.nodes;
  Plotly.restyle('lattice-view',{'marker.color':[$('lattice-view').data[1].x.map((_,i)=>i%count===index?'#dbab62':'#157d84')],'marker.size':[$('lattice-view').data[1].x.map((_,i)=>i%count===index?6:3.6)]},[1]);
}

async function selectCandidate(id, navigate=true) {
  try {
    const item=await getCandidate(id);state.current=item;persist();
    if(navigate)await goPage('design');
    $('candidate-name').textContent=pretty(item.name);
    $('candidate-origin').textContent=pretty(item.group).toUpperCase();
    $('evidence-badge').textContent=item.evidence;$('evidence-badge').classList.toggle('indigo',!!item.predictions);
    $('geometry-stats').innerHTML=`<div><strong>${item.nodes}</strong><span>Nodes</span></div><div><strong>${item.struts}</strong><span>Unique struts</span></div><div><strong>${item.components}</strong><span>Graph component${item.components===1?'':'s'}</span></div>`;
    $('geometry-table').innerHTML=`<p class="help">Lengths: ${item.lengths.map(fmt).join(' / ')} · model scale<br>Angles: ${item.angles.map(fmt).join('° / ')}°</p><label for="node-selector">Inspect node</label><select id="node-selector">${item.frac_coords.map((_,i)=>`<option value="${i}">Node ${i}</option>`).join('')}</select><p id="node-selection" role="status" class="help">Select a node here or in the 3D view.</p><div class="table-wrap"><table><caption>Fractional coordinates${item.nodes>100?' · first 100 nodes; export includes all nodes':''}</caption><thead><tr><th>Node</th><th>x</th><th>y</th><th>z</th><th>Neighbors</th></tr></thead><tbody>${item.frac_coords.slice(0,100).map((p,i)=>`<tr><td>${i}</td>${p.map(v=>`<td>${fmt(v)}</td>`).join('')}<td>${item.degrees[i]}</td></tr>`).join('')}</tbody></table></div>`;
    $('node-selector').onchange=event=>selectNode(Number(event.target.value));
    renderEvidence();renderStrip();renderProperties();restoreValidation();refreshSelectionButtons();
    if(state.view==='structure')await renderViewer();
  } catch(error){toast(error.message,true);}
}
function renderEvidence() {
  const item=state.current;if(!item)return;
  $('evidence-summary').innerHTML=`<div class="evidence-line"><span class="evidence-dot ${item.predictions?'indigo':''}"></span>${item.predictions?'Prediction available':'Geometry available'}</div><div class="evidence-line"><span class="evidence-dot amber"></span>Simulation not run</div>${item.predictions?`<div class="prediction-mini"><h3>Young's modulus</h3>${item.predictions.slice(0,3).map((v,i)=>`<div><span>Component ${i+1}</span><span class="meter"><i style="width:${Math.max(0,Math.abs(v)/Math.max(...item.predictions.slice(0,3).map(Math.abs),1e-12)*100)}%"></i></span><b>${fmt(v)}</b></div>`).join('')}<p class="help">Model scale · direction mapping unverified</p></div>`:`<div class="empty-evidence"><strong>Properties not recorded</strong><p>This saved lattice contains geometry, not measured elastic properties. Run the predictor to add model evidence.</p></div>`}<button class="button secondary full" data-predict ${state.capabilities.prediction?'':'disabled'}>${item.predictions?'Predict again':'Predict properties'} <span>↗</span></button>`;
  const checks=[[item.components===1,`${item.components===1?'Connected graph':item.components+' disconnected components'}`],
    [item.degrees.every(n=>n>=2),`${item.degrees.filter(n=>n<2).length} nodes with fewer than 2 neighbors`],
    [item.zero_length_struts===0,`${item.zero_length_struts} zero-length struts`]];
  $('geometry-checks').innerHTML=checks.map(([ok,text])=>`<div class="check-row"><span class="check-symbol ${ok?'':'warn'}">${ok?'✓':'!'}</span><span>${esc(text)}</span></div>`).join('')+item.warnings.map(text=>`<p class="footnote">${esc(text)}</p>`).join('');
  $('provenance').innerHTML=`<p class="source-name">${esc(item.source)}</p>${item.provenance?`<p class="help">${esc(item.provenance.kind)}${item.provenance.checkpoint?' · '+esc(item.provenance.checkpoint):''}</p>${item.provenance.supervisor_score!=null?`<p class="help">Supervisor score: ${fmt(item.provenance.supervisor_score)}. Separate from physical validation.</p>`:''}`:'<p class="help">Imported from existing research. Original prompt and iteration history are unavailable.</p>'}`;
}
function renderStrip() {
  const options=[state.current,...state.items.filter(x=>x.id!==state.current?.id&&!x.error)].filter(Boolean).slice(0,3);
  $('candidate-strip').innerHTML=options.map(item=>`<button class="candidate-mini ${item.id===state.current?.id?'active':''}" data-open="${item.id}" aria-label="Open ${esc(item.name)}">${thumbnail(item)}<span>${esc(pretty(item.name))}</span></button>`).join('');
}
function refreshSelectionButtons() {
  $('compare-count').textContent=state.compare.length;
  const current=state.current;
  $('save-button').disabled=!current;$('export-button').disabled=!current;
  $('save-button').textContent=current&&state.favorites.includes(current.id)?'★ Candidate saved':'☆ Save candidate';
  $('compare-add').textContent=current&&state.compare.includes(current.id)?'✓ Added to comparison':'＋ Add to comparison';
  $('compare-add').disabled=!current;
}
function toggleFavorite(id) {
  if(state.favorites.includes(id))state.favorites=state.favorites.filter(x=>x!==id);else state.favorites.push(id);
  persist();refreshSelectionButtons();if(state.page==='library')loadLibrary();
  toast(state.favorites.includes(id)?'Candidate saved to your shortlist.':'Candidate removed from your shortlist.');
}
function toggleCompare(id) {
  if(state.compare.includes(id))state.compare=state.compare.filter(x=>x!==id);
  else if(state.compare.length<4)state.compare.push(id);
  else {toast('Compare up to four candidates. Remove one to add another.');return;}
  persist();refreshSelectionButtons();
  if(state.page==='library')renderLibrary();if(state.page==='compare')renderCompare();
}

function renderProperties() {
  const item=state.current;if(!item)return;
  if(!item.predictions){$('property-content').innerHTML=`<div class="empty-state"><div class="empty-icon">∿</div><h3>Geometry is only the beginning.</h3><p>This file has no prediction vector. Run the local elastic-property predictor to inspect its response. No properties are inferred from the filename or conditioning field.</p><button class="button primary" data-predict ${state.capabilities.prediction?'':'disabled'}>Predict elastic properties →</button></div>`;return;}
  const groups=[['Young’s modulus',0,3],['Shear modulus',3,6],['Poisson ratios',6,12]];
  $('property-content').innerHTML=`<h3>Predicted elastic response</h3><p class="muted">Model scale; physical units and directional ordering are unverified. Component labels preserve the stored order.</p>${groups.map(([name,start,end])=>`<h3>${name}</h3><div class="property-chart" id="property-${start}"></div>`).join('')}<div class="table-wrap"><table><caption>Numerical values · prediction only</caption><thead><tr><th>Family</th><th>Component</th><th class="number">Value</th></tr></thead><tbody>${groups.map(([name,start,end])=>item.predictions.slice(start,end).map((v,i)=>`<tr><td>${name}</td><td>${i+1}</td><td class="number">${fmt(v)}</td></tr>`).join('')).join('')}</tbody></table></div>`;
  if(state.view==='properties')for(const [name,start,end] of groups){
    const values=item.predictions.slice(start,end),labels=values.map((_,i)=>'Component '+(i+1));
    Plotly.newPlot('property-'+start,[{type:'bar',orientation:'h',x:values,y:labels,marker:{color:values.map(v=>v<0?'#ae8dba':'#818bc5')},hovertemplate:'%{y}: %{x:.5g}<extra>Predicted</extra>'}],
      {height: end-start===6?260:185,margin:{l:85,r:20,t:15,b:40},paper_bgcolor:'white',plot_bgcolor:'white',font:{family:'system-ui',size:10,color:'#6e7c87'},xaxis:{zeroline:true,zerolinecolor:'#7d9197',gridcolor:'#edf1f2',title:{text:start===6?'Dimensionless':'Model scale'}},yaxis:{autorange:'reversed'},showlegend:false},{responsive:true,displayModeBar:false});
  }
}
async function setView(view) {
  state.view=view;
  for(const name of ['structure','properties','validate']){$(name+'-view').hidden=name!==view;$('tab-'+name).classList.toggle('active',name===view);$('tab-'+name).setAttribute('aria-selected',String(name===view));}
  if(view==='structure')await renderViewer();if(view==='properties')renderProperties();
}
function restoreValidation() {
  const form=$('validation-form');form.reset();
  const values=state.validation[state.current?.fingerprint];
  if(values)for(const [key,value] of Object.entries(values))if(form.elements[key])form.elements[key].value=value;
  $('validation-saved').textContent=values?'Saved setup for this exact geometry. Simulation has not been run.':'';
}

async function goPage(page) {
  if(!['design','library','compare','runs'].includes(page))page='design';
  state.page=page;document.querySelectorAll('.page').forEach(node=>node.hidden=node.id!=='page-'+page);
  document.querySelectorAll('.nav-button').forEach(node=>{node.classList.toggle('active',node.dataset.page===page);if(node.dataset.page===page)node.setAttribute('aria-current','page');else node.removeAttribute('aria-current');});
  history.replaceState(null,'','#'+page);
  if(page==='library')await loadLibrary();
  if(page==='compare')await renderCompare();
  if(page==='runs')await pollRuns(true);
  if(page==='design'&&state.current&&state.view==='structure')await renderViewer();
}
async function loadLibrary() {
  const request=++libraryRequest;
  try {
    if($('saved-only').checked) {
      const results=await Promise.allSettled(state.favorites.map(getCandidate));
      const items=results.filter(x=>x.status==='fulfilled').map(x=>x.value).filter(x=>(x.source+' '+x.name).toLowerCase().includes($('search').value.toLowerCase())&&(!$('group').value||x.group===$('group').value));
      if(request!==libraryRequest)return;
      state.total=items.length;state.library=items.slice(state.offset,state.offset+12);
    } else {
      const params=new URLSearchParams({q:$('search').value,group:$('group').value,offset:state.offset,limit:12});
      const result=await api('/candidates?'+params);
      if(request!==libraryRequest)return;
      cache(result.items);state.library=result.items;state.total=result.total;
      const value=$('group').value;$('group').innerHTML='<option value="">All collections</option>'+result.groups.map(group=>`<option value="${esc(group)}">${esc(pretty(group))}</option>`).join('');$('group').value=value;
    }
    renderLibrary();
  }catch(error){toast(error.message,true);}
}
function renderLibrary() {
  $('library-total').textContent=state.total.toLocaleString()+' lattices';
  $('library-grid').innerHTML=state.library.length?state.library.map(item=>`<article class="library-card ${state.compare.includes(item.id)?'selected':''}"><button class="library-open" data-open="${item.id}" ${item.error?'disabled':''}>${thumbnail(item)}<span class="card-body"><span class="card-title">${esc(pretty(item.name))}</span><span class="card-source">${esc(item.group||item.source)}</span>${item.error?`<span class="help">${esc(item.error)}</span>`:`<span class="card-metrics"><span><strong>${item.nodes}</strong> nodes</span><span><strong>${item.struts}</strong> struts</span><span>${esc(item.evidence)}</span></span>`}</span></button><div class="card-actions"><label class="check-label"><input type="checkbox" data-compare="${item.id}" ${state.compare.includes(item.id)?'checked':''} ${item.error?'disabled':''}>Compare</label><button class="save-small ${state.favorites.includes(item.id)?'saved':''}" data-save="${item.id}" aria-label="${state.favorites.includes(item.id)?'Unsave':'Save'} ${esc(item.name)}">${state.favorites.includes(item.id)?'★':'☆'}</button></div></article>`).join(''):'<div class="empty-state"><h3>No lattices match.</h3><p>Change the search or source collection, or save candidates from Design.</p></div>';
  $('page-info').textContent=state.total?`${state.offset+1}–${Math.min(state.offset+12,state.total)} of ${state.total.toLocaleString()}`:'No results';
  $('previous-page').disabled=state.offset===0;$('next-page').disabled=state.offset+12>=state.total;
}

async function renderCompare() {
  for(const node of document.querySelectorAll('.compare-view'))Plotly.purge(node);
  if(!state.compare.length){$('compare-content').innerHTML='<div class="panel empty-state"><div class="empty-icon">⇄</div><h3>Bring your options together.</h3><p>Add two to four lattices from Design or Library to compare their geometry and available predictions.</p><button data-page="library" class="button primary">Explore the library →</button></div>';return;}
  try {
    const results=await Promise.allSettled(state.compare.map(getCandidate));
    const items=results.filter(x=>x.status==='fulfilled').map(x=>x.value);
    if(items.length!==state.compare.length){state.compare=items.map(x=>x.id);persist();refreshSelectionButtons();toast('A missing candidate was removed from comparison.');}
    if(!items.length){renderCompare();return;}
    const span=Math.max(...items.map(item=>Math.max(...[0,1,2].map(i=>{const values=item.cart_coords.map(p=>p[i]);return Math.max(...values)-Math.min(...values);}))),...items.flatMap(item=>item.lengths));
    $('compare-content').innerHTML=`<div class="compare-grid" style="--columns:${items.length}">${items.map(item=>`<article class="compare-card"><div class="compare-card-head"><div><h3>${esc(pretty(item.name))}</h3><span class="badge ${item.predictions?'indigo':''}">${esc(item.evidence)}</span></div><button class="remove-button" data-remove="${item.id}" aria-label="Remove ${esc(item.name)}">×</button></div><div class="compare-view" id="compare-${item.id}" role="img" aria-label="Three-dimensional view of ${esc(item.name)}"></div><table><tbody><tr><th>Nodes</th><td class="number">${item.nodes}</td></tr><tr><th>Unique struts</th><td class="number">${item.struts}</td></tr><tr><th>Graph components</th><td class="number">${item.components}</td></tr><tr><th>Young’s · component 1</th><td class="number">${item.predictions?fmt(item.predictions[0]):'Not available'}</td></tr><tr><th>Simulation</th><td class="number">Not run</td></tr></tbody></table><button class="button secondary" data-open="${item.id}">Open in Design →</button></article>`).join('')}</div><div class="compare-chart-panel"><h3>Structural complexity</h3><p class="help">Node count and strut count are geometry measures, not relative density or stiffness.</p><div id="tradeoff-chart" class="tradeoff-chart"></div></div>`;
    let syncing=false;
    for(const item of items) {
      const plot=latticePlot(item,1,$('same-scale').checked?span:null);
      await Plotly.newPlot('compare-'+item.id,plot.data,plot.layout,plotConfig);
      $('compare-'+item.id).on('plotly_relayout',async event=>{
        if(syncing||!$('sync-camera').checked||!event['scene.camera'])return;
        syncing=true;
        await Promise.all(items.filter(x=>x.id!==item.id).map(x=>Plotly.relayout('compare-'+x.id,{'scene.camera':event['scene.camera']})));
        syncing=false;
      });
    }
    await Plotly.newPlot('tradeoff-chart',[{type:'scatter',mode:'markers+text',x:items.map(x=>x.nodes),y:items.map(x=>x.struts),text:items.map((_,i)=>String.fromCharCode(65+i)),textposition:'top center',customdata:items.map(x=>x.name),marker:{size:12,color:'#168a8f',line:{color:'white',width:2}},hovertemplate:'%{customdata}<br>%{x} nodes · %{y} struts<extra></extra>'}],{margin:{l:50,r:25,t:20,b:45},font:{family:'system-ui',size:11,color:'#6e7c87'},paper_bgcolor:'white',plot_bgcolor:'white',xaxis:{title:{text:'Node count'},gridcolor:'#edf1f2',rangemode:'tozero',dtick:Math.max(1,Math.ceil(Math.max(...items.map(x=>x.nodes))/8))},yaxis:{title:{text:'Unique struts'},gridcolor:'#edf1f2',rangemode:'tozero'},showlegend:false},{responsive:true,displayModeBar:false});
  }catch(error){toast(error.message,true);}
}

async function predict() {
  if(!state.current)return;
  await submitRun({kind:'prediction',candidate_id:state.current.id,seed:Number($('seed').value)});
}
async function submitRun(payload) {
  if(submitting)return;
  submitting=true;
  const buttons=document.querySelectorAll('[data-predict],#generate-button');buttons.forEach(x=>x.disabled=true);
  try {
    await api('/runs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    $('review-dialog').close();toast('Run started. You can keep exploring while it works.');
    await pollRuns(true);
  }catch(error){toast(error.message,true);}finally{submitting=false;renderEvidence();$('generate-button').disabled=!state.capabilities.generation;}
}
async function pollRuns(force=false) {
  if(polling)return;polling=true;
  try {
    const runs=await api('/runs');state.runs=runs;
    const active=runs.find(x=>['queued','running'].includes(x.state));
    $('run-dot').hidden=!active;$('active-run').hidden=!active;
    if(active)$('active-run').innerHTML=`<span>${esc(active.kind==='prediction'?'Prediction':'Generation')} · ${esc(active.stage)}</span><a href="#runs" data-page="runs">Follow run →</a>`;
    const signature=JSON.stringify(runs);
    if(force||signature!==runSignature){
      $('runs-list').innerHTML=runs.length?runs.map(run=>`<article class="run-card"><div><h3>${run.kind==='prediction'?'Elastic property prediction':'Lattice generation'}</h3>${run.prompt?`<p>${esc(run.prompt)}</p>`:''}<p class="${run.state==='failed'?'run-error':''}">${esc(run.stage)}</p><span class="run-date">${new Date(run.created*1000).toLocaleString()} · ${run.candidates.length} saved candidate${run.candidates.length===1?'':'s'}</span></div><div class="run-actions"><span class="badge ${run.state==='failed'?'amber':run.state==='complete'?'':'indigo'}">${esc(({complete:'Completed',failed:'Failed',queued:'Queued',running:'Running',interrupted:'Interrupted'})[run.state]||run.state)}</span>${run.candidates.length?`<button class="button secondary" data-open="${run.selected_candidate||run.candidates.at(-1)}">Open result →</button>`:''}</div></article>`).join(''):'<div class="panel empty-state"><div class="empty-icon">◷</div><h3>Your exploration, recorded.</h3><p>Prediction and generation runs will appear here. Existing saved lattices remain available in the Library.</p><button class="button primary" data-page="design">Back to Design →</button></div>';
      if(runSignature&&signature!==runSignature){const done=runs.find(x=>x.state==='complete'&&!JSON.parse(runSignature).some(old=>old.id===x.id&&old.state==='complete'));if(done)toast('Your '+done.kind+' is complete. Open its result in Runs.');}
      runSignature=signature;
    }
  }catch(error){if(force)toast(error.message,true);}finally{polling=false;}
}
function downloadText(name,text,type='text/plain') {
  const url=URL.createObjectURL(new Blob([text],{type})),link=document.createElement('a');link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
const csvCell=value=>{let text=String(value??'');if(/^[=+@\t\r]/.test(text)||(/^-/ .test(text)&&!Number.isFinite(Number(text))))text="'"+text;return '"'+text.replaceAll('"','""')+'"';};
function exportCSV(items,name) {
  const headers=['Name','Evidence','Source','Nodes','Unique struts','Graph components',...Array.from({length:12},(_,i)=>`Predicted component ${i+1}`),'Units','Simulation'];
  const rows=items.map(item=>[item.name,item.evidence,item.source,item.nodes,item.struts,item.components,...(item.predictions||Array(12).fill('Not available')),'Model scale; mapping unverified','Not run']);
  downloadText(name,[headers,...rows].map(row=>row.map(csvCell).join(',')).join('\r\n'),'text/csv');
}
async function exportCandidate(kind) {
  const item=state.current;if(!item)return;
  if(kind==='npz'){const a=document.createElement('a');a.href='/api/candidates/'+item.id+'/download';a.download=item.name+'.npz';a.click();}
  if(kind==='csv')exportCSV([item],item.name+'.csv');
  if(kind==='report')downloadText(item.name+'-report.json',JSON.stringify({title:'MetaSymbO design report',exported:new Date().toISOString(),draft_brief:state.brief,candidate:item,validation_setup:state.validation[item.fingerprint]||null,simulation:'Not run',note:'Draft brief may differ from the original generation prompt. Model units and directional mapping are unverified.'},null,2),'application/json');
  if(kind==='png'){await setView('structure');await Plotly.downloadImage('lattice-view',{format:'png',filename:item.name,width:1200,height:900});}
  $('export-dialog').close();
}
async function refreshCapabilities() {
  state.capabilities=await api('/status');
  $('footer-status').textContent=state.capabilities.count.toLocaleString()+' project lattices · '+(state.capabilities.public_demo?'Public research demo':'Local workspace');
  $('public-note').hidden=!state.capabilities.public_demo;
  $('brief').maxLength=state.capabilities.max_prompt;
  $('attempts').max=state.capabilities.max_attempts;
  $('generation-hint').textContent=state.capabilities.busy?'The demo is busy. You can explore saved designs while a run finishes.':state.capabilities.generation?'Generation is available. A reviewed brief starts a new run.':'Generation is temporarily unavailable. Explore saved designs and try again later.';
  $('setup-content').innerHTML=[['Geometry library',true],['Model environment',state.capabilities.model_runtime],['Checkpoints',state.capabilities.checkpoints],['Reference dataset',state.capabilities.dataset],['API credential configured',state.capabilities.api_key_configured],['Simulation integration',false]].map(([name,ready])=>`<div class="setup-row"><span>${name}</span><span class="badge ${ready?'':'amber'}">${ready?'Ready':'Not available'}</span></div>`).join('');
}

document.addEventListener('click',async event=>{
  const target=event.target.closest('[data-page],[data-view],[data-open],[data-save],[data-remove],[data-predict],[data-export],[data-preset],[data-camera]');
  if(!target||target.disabled)return;
  try {
    if(target.dataset.page){event.preventDefault();await goPage(target.dataset.page);}
    if(target.dataset.view)await setView(target.dataset.view);
    if(target.dataset.open)await selectCandidate(target.dataset.open);
    if(target.dataset.save)toggleFavorite(target.dataset.save);
    if(target.dataset.remove)toggleCompare(target.dataset.remove);
    if('predict' in target.dataset)await predict();
    if(target.dataset.export)await exportCandidate(target.dataset.export);
    if(target.dataset.preset){$('brief').value=target.dataset.preset;$('brief').dispatchEvent(new Event('input'));}
    if(target.dataset.camera){const eye={xy:{x:0,y:0,z:2.4},xz:{x:0,y:-2.4,z:0},yz:{x:2.4,y:0,z:0}}[target.dataset.camera];await Plotly.relayout('lattice-view',{'scene.camera':{eye,up:target.dataset.camera==='xy'?{x:0,y:1,z:0}:{x:0,y:0,z:1},projection:{type:'orthographic'}}});}
  }catch(error){toast(error.message,true);}
});
document.addEventListener('change',event=>{if(event.target.dataset.compare)toggleCompare(event.target.dataset.compare);});
document.querySelector('.local-tabs').addEventListener('keydown',event=>{
  if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;
  event.preventDefault();const names=['structure','properties','validate'],index=names.indexOf(state.view);
  const next=event.key==='Home'?0:event.key==='End'?2:(index+(event.key==='ArrowRight'?1:2))%3;
  setView(names[next]);$('tab-'+names[next]).focus();
});
$('import-button').onclick=()=>$('import-file').click();
$('new-design-button').onclick=()=>{$('brief').scrollIntoView({block:'center',behavior:'smooth'});$('brief').focus({preventScroll:true});};
$('brief').value=state.brief;
$('brief').addEventListener('input',()=>{state.brief=$('brief').value;persist();$('brief-state').textContent='Draft saved locally. Review this revision before generating.';});
$('review-button').onclick=async()=>{
  if(!state.brief.trim()){toast('Describe your design goal first.');$('brief').focus();return;}
  if(!$('attempts').reportValidity()||!$('seed').reportValidity())return;
  try{await refreshCapabilities();}catch(error){toast(error.message,true);return;}
  $('review-text').textContent=state.brief;$('review-settings').textContent=`${$('logic').selectedOptions[0].text} · Up to ${$('attempts').value} attempts · Seed ${$('seed').value}`;
  const unsupported=/thermal|acoustic|fatigue|impact|printab|manufactur|conductiv|radiation|magnet/i.test(state.brief);
  $('review-warning').textContent=!state.capabilities.generation?$('generation-hint').textContent:unsupported?'This brief includes a goal outside the elastic predictor’s scope. Those goals will remain unevaluated.':'Goals guide generation; they are not guaranteed constraints. The selected lattice is a reference, not a fixed generation seed.';
  $('generate-button').disabled=!state.capabilities.generation||state.capabilities.busy;$('review-dialog').showModal();
};
$('review-edit').onclick=()=>{$('review-dialog').close();$('brief').focus();};
$('generate-button').onclick=()=>submitRun({kind:'generation',prompt:state.brief,logic_mode:$('logic').value,attempts:Number($('attempts').value),seed:Number($('seed').value)});
$('save-button').onclick=()=>{if(state.current)toggleFavorite(state.current.id);};
$('compare-add').onclick=()=>{if(state.current)toggleCompare(state.current.id);};
$('export-button').onclick=()=>{$('export-dialog').showModal();};
$('setup-button').onclick=async()=>{try{await refreshCapabilities();$('setup-dialog').showModal();}catch(error){toast(error.message,true);}};
$('repeat').onchange=()=>renderViewer();
$('reset-view').onclick=()=>Plotly.relayout('lattice-view',{'scene.camera':{eye:{x:.9,y:1.05,z:.7},up:{x:0,y:0,z:1},projection:{type:'perspective'}}});
$('search').oninput=()=>{clearTimeout(searchTimer);searchTimer=setTimeout(()=>{state.offset=0;loadLibrary();},220);};
for(const id of ['group','saved-only'])$(id).onchange=()=>{state.offset=0;loadLibrary();};
$('previous-page').onclick=()=>{state.offset=Math.max(0,state.offset-12);loadLibrary();};
$('next-page').onclick=()=>{state.offset+=12;loadLibrary();};
$('same-scale').onchange=()=>renderCompare();
$('refresh-runs').onclick=()=>pollRuns(true);
$('export-comparison').onclick=async()=>{try{if(!state.compare.length){toast('Add candidates to comparison first.');return;}exportCSV(await Promise.all(state.compare.map(getCandidate)),'metasymbo-comparison.csv');}catch(error){toast(error.message,true);}};
$('validation-form').onsubmit=event=>{
  event.preventDefault();if(!state.current)return;
  const values=Object.fromEntries(new FormData(event.currentTarget));state.validation[state.current.fingerprint]=values;persist();
  $('validation-saved').textContent='Saved locally for this geometry. Simulation has not been run.';toast('Validation assumptions saved.');
};
$('import-file').onchange=async event=>{
  const file=event.target.files[0];if(!file)return;
  try{if(file.size>4*1024*1024)throw new Error('Choose an NPZ file smaller than 4 MB.');const item=await api('/import',{method:'POST',headers:{'Content-Type':'application/octet-stream'},body:file});cache([item]);state.items.unshift(item);await selectCandidate(item.id);toast('Lattice imported. Original history may be unavailable.');await refreshCapabilities();}catch(error){toast(error.message,true);}finally{event.target.value='';}
};

async function start() {
  try {
    if(!window.Plotly)throw new Error('The local Plotly bundle could not load. Refresh the workbench.');
    await refreshCapabilities(); // Establish the session before parallel private API requests.
    const result=await api('/candidates?limit=12');
    state.items=result.items;cache(result.items);
    let initial=state.selected;
    if(initial){try{await getCandidate(initial);}catch{initial=null;}}
    initial=initial||result.items.find(x=>!x.error)?.id;
    if(initial)await selectCandidate(initial,false);
    else {$('candidate-name').textContent='No saved lattices yet';$('lattice-view').innerHTML='<div class="empty-state"><h3>Bring a lattice into view.</h3><p>Import an NPZ file, or review a brief to generate your first candidate.</p></div>';$('evidence-summary').innerHTML='<p class="muted">Select or import a lattice to inspect its evidence.</p>';}
    refreshSelectionButtons();await goPage(location.hash.slice(1)||'design');await pollRuns();
    setInterval(()=>pollRuns(),4000);
  }catch(error){toast(error.message,true);$('candidate-name').textContent='Workspace could not load';$('lattice-view').innerHTML='<div class="empty-state"><p>'+esc(error.message)+'</p></div>';}
}
start();
