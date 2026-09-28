"use strict";
const $ = id => document.getElementById(id);
let runs = [], episodes = [];
function label(text, className) { const node=document.createElement("div");node.textContent=text;node.className=className||"";return node; }
async function get(path) { const response=await fetch(path,{cache:"no-store"});if(!response.ok)throw new Error(`HTTP ${response.status}`);return response.json(); }
function showEpisode(episode) {
  $("episode-title").textContent=`에피소드 #${episode.index}`;
  $("episode-detail").textContent=`${episode.frames} 프레임 · ${episode.duration_s}초`+(episode.tasks?.length?` · ${episode.tasks.join(" / ")}`:"");
  const holder=$("episode-videos");holder.replaceChildren();
  if(!episode.videos.length){holder.append(label("저장된 영상이 없습니다.","episode-empty"));return;}
  for(const item of episode.videos){
    const section=document.createElement("section");section.append(label(item.key.replace("observation.images.",""),"episode-muted"));
    const video=document.createElement("video");video.controls=true;video.preload="metadata";video.className="episode-video";video.src=item.url;
    video.addEventListener("loadedmetadata",()=>{video.currentTime=item.start_s;});
    video.addEventListener("timeupdate",()=>{if(video.currentTime>=item.end_s){video.pause();video.currentTime=item.start_s;}});
    section.append(video);holder.append(section);
  }
}
function renderEpisodes() {
  $("episode-count").textContent=`(${episodes.length})`;
  const list=$("episode-list");list.replaceChildren();
  if(!episodes.length){list.append(label("저장된 에피소드가 없습니다. 폐기된 시도는 데이터셋에 포함되지 않습니다.","episode-empty"));$("episode-title").textContent="에피소드 없음";$("episode-detail").textContent="";$("episode-videos").replaceChildren();return;}
  for(const episode of episodes){const button=document.createElement("button");button.type="button";button.className="episode-item";button.textContent=`#${episode.index} · ${episode.duration_s}초 · ${episode.frames} 프레임`;button.addEventListener("click",()=>{list.querySelectorAll("button").forEach(item=>item.removeAttribute("aria-current"));button.setAttribute("aria-current","true");showEpisode(episode);});list.append(button);}
  list.firstChild.click();
}
async function selectRun() {
  const run=runs.find(item=>item.id===$("run-select").value);if(!run)return;
  $("run-detail").textContent=`${run.episodes}개 저장 · ${run.frames} 프레임 · ${run.fps} FPS`+(Object.keys(run.discard_reasons).length?` · 폐기: ${Object.entries(run.discard_reasons).map(([k,v])=>`${k} ${v}`).join(", ")}`:"");
  try{episodes=(await get(`/api/episodes/${encodeURIComponent(run.id)}`)).episodes;renderEpisodes();}catch(error){$("episode-list").replaceChildren(label(`불러오기 실패: ${error.message}`,"episode-empty"));}
}
async function refresh(){
  const previous=$("run-select").value;
  try{runs=(await get("/api/episodes")).runs;const select=$("run-select");select.replaceChildren();
    for(const run of runs){const option=document.createElement("option");option.value=run.id;option.textContent=`${run.id} (${run.episodes}개)`;select.append(option);}
    if(!runs.length){$("run-detail").textContent="아직 수집 데이터셋이 없습니다.";$("episode-list").replaceChildren();return;}
    if(runs.some(run=>run.id===previous))select.value=previous;
    await selectRun();
  }catch(error){$("run-detail").textContent=`불러오기 실패: ${error.message}`;}
}
$("run-select").addEventListener("change",selectRun);$("refresh").addEventListener("click",refresh);refresh();
