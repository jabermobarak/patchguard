const $ = (id) => document.getElementById(id);
let result = null;
let currentObjectUrl = null;
let busy = false;

function setBusy(value) {
  busy = value;
  document.querySelectorAll('.sample').forEach(button => { button.disabled = value; });
  $('file').disabled = value;
  $('state').textContent = value ? 'Inspecting…' : 'Ready';
}

async function inspect(blob, name) {
  if (busy) return;
  setBusy(true);
  result = null;
  $('download').disabled = true;
  $('overlay').removeAttribute('src');
  for (const id of ['decision', 'ratio', 'regions', 'latency']) $(id).textContent = '—';
  $('details').textContent = 'Inspecting…';
  if (currentObjectUrl) URL.revokeObjectURL(currentObjectUrl);
  currentObjectUrl = URL.createObjectURL(blob);
  $('original').src = currentObjectUrl;
  $('original').alt = name;
  const form = new FormData();
  form.append('file', blob, name);
  try {
    const response = await fetch('/api/inspect', { method: 'POST', body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Inspection failed');
    result = data;
    $('overlay').src = data.overlay;
    $('decision').textContent = {normal:'Normal', anomaly:'Anomaly', review:'Review'}[data.status];
    $('decision').className = data.status;
    $('ratio').textContent = `${data.score_ratio.toFixed(2)}×`;
    $('regions').textContent = data.regions.length;
    $('latency').textContent = `${Math.round(data.latency_ms)} ms`;
    $('marker').style.left = `${Math.min(data.score_ratio / 2, 1) * 99.5}%`;
    $('explanation').textContent = (data.brightness_shift || data.texture_shift)
      ? 'Brightness or texture sharpness differs from calibration images. Review the capture conditions before trusting the decision.'
      : { normal: 'The surface is within the normal score range.',
          anomaly: 'Unusual appearance detected. Warm regions show where the surface differs from normal.',
          review: 'The score is close to the decision threshold. A person should inspect this image.' }[data.status];
    const {overlay, patch_scores, ...summary} = data;
    $('details').textContent = JSON.stringify(summary, null, 2);
    $('download').disabled = false;
  } catch (error) {
    $('decision').textContent = 'Error';
    $('explanation').textContent = error.message;
    $('details').textContent = 'No valid result.';
  } finally { setBusy(false); }
}

async function selectSample(kind, button) {
  if (busy) return;
  try {
    const response = await fetch(`/api/examples/${kind}`);
    if (!response.ok) throw new Error('Could not load sample');
    document.querySelectorAll('.sample').forEach(item => item.classList.remove('selected'));
    button.classList.add('selected');
    await inspect(await response.blob(), `${kind}.png`);
  } catch (error) { $('explanation').textContent = error.message; }
}

for (const kind of ['normal', 'scratch', 'stain', 'hole']) {
  const button = document.createElement('button');
  button.className = 'sample';
  button.setAttribute('aria-label', `Inspect ${kind} example`);
  const image = document.createElement('img');
  image.src = `/api/examples/${kind}`;
  image.alt = '';
  button.append(image, document.createTextNode(kind[0].toUpperCase() + kind.slice(1)));
  button.addEventListener('click', () => selectSample(kind, button));
  $('samples').append(button);
}
$('file').addEventListener('change', () => {
  const file = $('file').files[0];
  if (!file) return;
  if (file.size > 8*1024*1024) { $('explanation').textContent = 'Use an image smaller than 8 MiB.'; return; }
  document.querySelectorAll('.sample').forEach(item => item.classList.remove('selected'));
  inspect(file, file.name);
});
$('download').addEventListener('click', () => {
  if (!result) return;
  const {overlay, ...data} = result;
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], {type:'application/json'}));
  const a = document.createElement('a'); a.href = url; a.download = 'patchguard-inspection.json'; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
fetch('/api/model').then(response => { if (!response.ok) throw new Error(); return response.json(); })
  .then(data => { $('model-id').textContent = data.model_id;
    $('model-features').textContent = `${data.scoring} · CPU inference`;
    selectSample('scratch', $('samples').children[1]); })
  .catch(() => { $('model-id').textContent = 'Unavailable'; $('explanation').textContent = 'Model service is unavailable.'; });
