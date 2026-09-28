"use strict";
// Preview-only interactions. No robot or camera API is contacted.
const theme=document.getElementById("theme-toggle");
theme?.addEventListener("click",()=>{
  const dark=!document.documentElement.classList.contains("dark");
  document.documentElement.classList.toggle("dark",dark);
  document.documentElement.style.colorScheme=dark?"dark":"light";
  try{localStorage.setItem("so101-theme",dark?"dark":"light");}catch(_){}
});
for(const button of document.querySelectorAll("[data-command-tab]"))button.addEventListener("click",()=>{
  for(const tab of document.querySelectorAll("[data-command-tab]")){
    const selected=tab===button;tab.setAttribute("aria-selected",String(selected));
    document.getElementById(tab.dataset.commandTab).hidden=!selected;
  }
});
for(const button of document.querySelectorAll("[data-diagnostic-tab]"))button.addEventListener("click",()=>{
  const value=button.dataset.diagnosticTab;
  document.body.dataset.diagnosticTab=value;
  for(const tab of document.querySelectorAll("[data-diagnostic-tab]"))tab.setAttribute("aria-selected",String(tab===button));
  document.getElementById("robot-diagnostics").hidden=["detections","history","episodes"].includes(value);
  document.getElementById("diagnostic-detections").hidden=value!=="detections";
  document.getElementById("diagnostic-history").hidden=value!=="history";
  document.getElementById("diagnostic-episodes").hidden=value!=="episodes";
});
if(new URLSearchParams(location.search).get("tab")==="episodes")document.querySelector("[data-diagnostic-tab=episodes]").click();
document.getElementById("composer")?.addEventListener("submit",event=>event.preventDefault());
