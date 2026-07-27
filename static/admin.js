document.addEventListener("DOMContentLoaded", () => {
  const body = document.getElementById("feedback-body");
  const count = document.getElementById("correction-count");
  const state = document.getElementById("training-state");
  const gate = document.getElementById("candidate-gate");
  const active = document.getElementById("active-model");
  const message = document.getElementById("admin-message");
  const metricsGrid = document.getElementById("metrics-grid");
  const metricsJson = document.getElementById("metrics-json");
  const empty = document.getElementById("empty-state");
  const search = document.getElementById("search-input");
  const promoteBtn = document.getElementById("promote-btn");
  const retrainBtn = document.getElementById("retrain-btn");
  let rows = [];

  const labels = {not_sarcastic:"Not sarcastic",sarcastic:"Sarcastic",insufficient_context:"Insufficient context"};
  function setMessage(text, kind="") { message.textContent=text; message.className=`admin-message ${kind}`; }
  async function api(url, options={}) { const response=await fetch(url,{headers:{"Content-Type":"application/json"},...options}); const data=await response.json().catch(()=>({})); if(!response.ok) throw new Error(typeof data.detail==="string"?data.detail:`Request failed (${response.status}).`); return data; }
  function pct(value){ return typeof value==="number"?`${(value*100).toFixed(1)}%`:"—"; }
  function escapeText(value){ const node=document.createElement("span"); node.textContent=value??""; return node.innerHTML; }

  function renderRows(){
    const term=search.value.trim().toLowerCase();
    const filtered=rows.filter(row=>!term || Object.values(row).some(v=>String(v).toLowerCase().includes(term)));
    body.innerHTML=""; empty.hidden=filtered.length!==0;
    for(const row of filtered){
      const tr=document.createElement("tr");
      tr.innerHTML=`<td>${row.id}</td><td class="text-cell"><textarea class="edit-text" rows="3">${escapeText(row.text)}</textarea></td><td>${escapeText(row.predicted_label)}</td><td>${pct(row.predicted_score)}</td><td><select><option value="not_sarcastic">Not sarcastic</option><option value="sarcastic">Sarcastic</option><option value="insufficient_context">Insufficient context</option></select></td><td>${escapeText(row.source)}</td><td>${escapeText(row.timestamp)}</td><td><div class="row-actions"><button class="save-btn">Save</button><button class="delete-btn">Delete</button></div></td>`;
      tr.querySelector("select").value=row.correct_label;
      tr.querySelector(".save-btn").addEventListener("click",()=>saveRow(row.id,tr));
      tr.querySelector(".delete-btn").addEventListener("click",()=>deleteRow(row.id));
      body.appendChild(tr);
    }
  }

  async function saveRow(id,tr){ try{ const text=tr.querySelector("textarea").value.trim(); const correct_label=tr.querySelector("select").value; await api(`/admin/feedback/${id}`,{method:"PUT",body:JSON.stringify({text,correct_label})}); setMessage("Correction updated.","success"); await loadFeedback(); }catch(e){setMessage(e.message,"error");} }
  async function deleteRow(id){ if(!confirm("Delete this correction?")) return; try{ await api(`/admin/feedback/${id}`,{method:"DELETE"}); setMessage("Correction deleted.","success"); await loadFeedback(); }catch(e){setMessage(e.message,"error");} }
  async function loadFeedback(){ const data=await api("/admin/feedback"); rows=data.items||[]; count.textContent=data.count||0; renderRows(); }

  function addMetric(name,value){ const div=document.createElement("div"); div.className="metric"; div.innerHTML=`<small>${name}</small><strong>${value}</strong>`; metricsGrid.appendChild(div); }
  function renderMetrics(metrics){ metricsGrid.innerHTML=""; metricsJson.textContent=metrics&&Object.keys(metrics).length?JSON.stringify(metrics,null,2):"No candidate metrics available."; const test=metrics?.test_metrics||{}; const real=metrics?.real_world_gate_metrics||{}; addMetric("Test accuracy",pct(test.accuracy)); addMetric("Test F1",pct(test.f1)); addMetric("ROC-AUC",typeof test.roc_auc==="number"?test.roc_auc.toFixed(3):"—"); addMetric("Real-world accuracy",pct(real.accuracy)); addMetric("Real-world specificity",pct(real.specificity)); }

  async function loadStatus(){ const data=await api("/admin/status"); const s=data.retraining_status||{}; state.textContent=s.state||"idle"; state.className=s.state==="failed"?"state-error":s.state==="passed"||s.state==="promoted"?"state-pass":"state-warn"; const candidate=data.candidate||{}; gate.textContent=candidate.promotion_gate_passed?"Passed":candidate.metrics_available?"Not passed":"Not available"; gate.className=candidate.promotion_gate_passed?"state-pass":candidate.metrics_available?"state-error":"state-warn"; active.textContent=data.active_model?.model_version||"Unknown"; promoteBtn.disabled=!candidate.promotion_gate_passed; retrainBtn.disabled=!data.retraining_available; renderMetrics(candidate.metrics||{}); }

  async function retrain(){ if(!confirm("Start candidate retraining using saved corrections?")) return; retrainBtn.disabled=true; try{ const data=await api("/admin/retrain",{method:"POST",body:"{}"}); setMessage(data.message,"success"); await loadStatus(); }catch(e){setMessage(e.message,"error");} finally{setTimeout(loadStatus,1000);} }
  async function promote(){ if(!confirm("Promote the passing candidate? The current active model will be backed up and Uvicorn must be restarted.")) return; try{ const data=await api("/admin/promote",{method:"POST",body:JSON.stringify({confirmation:"PROMOTE"})}); setMessage(data.message,"success"); await loadStatus(); }catch(e){setMessage(e.message,"error");} }
  async function refresh(){ try{ await Promise.all([loadFeedback(),loadStatus()]); setMessage("Administration data refreshed."); }catch(e){setMessage(e.message,"error");} }

  search.addEventListener("input",renderRows);
  document.getElementById("refresh-btn").addEventListener("click",refresh);
  retrainBtn.addEventListener("click",retrain);
  promoteBtn.addEventListener("click",promote);
  refresh();
  setInterval(loadStatus,5000);
});