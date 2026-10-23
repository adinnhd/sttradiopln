import { visibleDevices } from './devices.mjs';
import { WaveformView } from './waveform.mjs';

const $ = (id) => document.getElementById(id);
const waveform = new WaveformView($('waveform'));
let allDevices = [], devices = [], filter = 'speech', busy = false, connected = false;
let page = 1, selectedDate = '', listLoading = true;
let listSignature = '', lastConfig = null, lastStatus = null;
const labels = {speech:'Percakapan', uncertain:'Perlu diperiksa', 'non-speech':'Tanpa percakapan', pending:'Sedang diperiksa'};
const stateLabels = {stopped:'Tidak aktif', listening:'Mendengarkan', recording:'Merekam', waiting:'Menunggu hening', error:'Terhenti'};
function notice(message='') { $('notice').textContent = message; $('notice').hidden = !message; }
async function api(path, options={}) {
  const response = await fetch(path, {cache:'no-store', ...options});
  const body = await response.json();
  if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : 'Permintaan gagal. Periksa pilihan perangkat.');
  return body;
}
function controls() {
  const active = lastStatus && ['listening','recording','waiting'].includes(lastStatus.state);
  $('start').disabled = busy || !connected || active || !$('device').value;
  $('stop').disabled = busy || !connected || !active;
  $('device').disabled = busy || active;
  $('channels').disabled = busy || active;
  $('refresh').disabled = busy || active;
  $('all-drivers').disabled = busy || active;
}
function deviceHint() {
  const device = devices.find(d => String(d.id) === $('device').value);
  const stereoOption = [...$('channels').options].find(option => option.value === '2');
  stereoOption.disabled = !device || device.max_channels < 2;
  if (stereoOption.disabled && $('channels').value === '2') $('channels').value = 'auto';
  const automatic = $('channels').value === 'auto';
  const channels = device ? (automatic ? Math.min(device.max_channels, 2) : Number($('channels').value)) : null;
  $('device-hint').textContent = device
    ? `${device.name} · ${automatic ? 'Otomatis' : 'Manual'}: ${channels} kanal`
    : devices.length
      ? 'Pilih input yang terhubung ke radio. Jika belum muncul, buka Pengaturan audio.'
      : 'Belum ada perangkat input. Sambungkan perangkat, lalu klik Perbarui.';
  controls();
}
async function loadDevices() {
  try {
    allDevices = (await api('/api/devices')).items;
    populateDevices();
  } catch (error) { notice(error.message); }
}
function populateDevices() {
  const previous = $('device').value;
  devices = visibleDevices(allDevices, $('all-drivers').checked);
  $('device').replaceChildren(new Option('Pilih perangkat audio...', ''));
  devices.forEach(d => {
    const duplicate = devices.some(other => other.id !== d.id && other.name === d.name);
    $('device').add(new Option(duplicate ? `${d.name} (${d.hostapi})` : d.name, String(d.id)));
  });
  if (devices.some(d => String(d.id) === previous)) $('device').value = previous;
  if ($('device').value !== previous) $('channels').value = 'auto';
  deviceHint();
}
function renderStatus(s) {
  lastStatus = s;
  const active = ['listening','recording','waiting'].includes(s.state);
  waveform.update(s.waveform, active);
  if (active && devices.some(device => device.id === s.device_id)) {
    $('device').value = String(s.device_id);
    $('channels').value = s.channel_mode === 'auto' ? 'auto' : String(s.channels);
    deviceHint();
  }
  const state = $('state'); state.dataset.state = s.state;
  state.replaceChildren(document.createElement('i'), document.createTextNode(stateLabels[s.state] || s.state));
  $('level').textContent = s.level_db <= -120 ? '—' : s.level_db.toFixed(1);
  const seconds = Math.floor(s.duration);
  $('duration').textContent = `${String(Math.floor(seconds/60)).padStart(2,'0')}:${String(seconds%60).padStart(2,'0')}`;
  const messages = {
    stopped:['Monitoring belum dimulai','Pilih perangkat audio, lalu mulai monitoring.'],
    listening:['Menunggu komunikasi','Audio input aktif. Rekaman dipicu otomatis.'],
    recording:['Komunikasi terdeteksi','Audio sedang disimpan ke WAV.'],
    waiting:['Menunggu akhir komunikasi',`Hening ${s.silence_elapsed.toFixed(1)} / ${s.config.silence_seconds} detik`],
    error:['Input audio terhenti','Periksa koneksi perangkat lalu mulai kembali.']
  };
  const message = messages[s.state] || messages.stopped;
  $('activity-title').textContent = message[0]; $('activity-detail').textContent = message[1];
  $('queue-status').hidden = !s.pending;
  $('queue-status').textContent = s.pending ? `${s.pending} rekaman sedang diperiksa` : '';
  const config = JSON.stringify([s.config,s.vad_config]);
  if (config !== lastConfig) {
    $('threshold').textContent = $('cfg-threshold').textContent = `${s.config.threshold_db} dBFS`;
    $('cfg-prebuffer').textContent = `${s.config.prebuffer_seconds} detik`;
    $('cfg-silence').textContent = `${s.config.silence_seconds} detik`;
    $('cfg-vad').textContent = `${s.vad_config.min_speech_duration_ms} ms`;
    lastConfig = config;
  }
  if (s.error) notice(s.error);
  else if (s.overflows) notice(`Audio input mengalami ${s.overflows} overflow; sebagian audio mungkin terlewat. Periksa perangkat dan beban laptop.`);
  controls();
}
function element(tag, text, className) {
  const node = document.createElement(tag); node.textContent = text;
  if (className) node.className = className;
  return node;
}
function renderRecordings(data) {
  for (const key of ['speech','uncertain','non-speech']) $(`count-${key}`).textContent = data.counts[key] || 0;
  $('total-count').textContent = Object.values(data.counts).reduce((sum,n) => sum+n,0);
  const signature = JSON.stringify([filter,selectedDate,data.page,data.total,data.items]);
  // Jangan mengganti elemen audio yang sedang diputar saat polling.
  if ([...document.querySelectorAll('audio')].some(a => !a.paused)) return;
  page = data.page;
  listLoading = false;
  const from = data.total ? (page-1)*data.page_size+1 : 0;
  const to = data.total ? from+data.items.length-1 : 0;
  const summary = data.total
    ? `Menampilkan ${from}–${to} dari ${data.total} rekaman`
    : '0 rekaman sesuai filter';
  if ($('pagination-summary').textContent !== summary) $('pagination-summary').textContent = summary;
  $('page-status').textContent = data.total ? `Halaman ${page} dari ${data.total_pages}` : '';
  $('previous-page').disabled = page <= 1;
  $('next-page').disabled = page >= data.total_pages;
  if (signature === listSignature) return;
  listSignature = signature;
  const list = $('recording-list'); list.replaceChildren();
  if (!data.items.length) {
    const empty = element('div','','empty');
    empty.append(element('h3', filter === 'speech' ? 'Belum ada percakapan' : 'Belum ada rekaman pada filter ini'),element('p',filter === 'speech' ? 'Rekaman percakapan akan muncul setelah komunikasi selesai diperiksa.' : 'Pilih filter lain atau mulai monitoring untuk menambahkan rekaman.'));
    list.append(empty); return;
  }
  for (const item of data.items) {
    const row = element('article','','recording'), details = element('div','');
    row.dataset.status = item.status;
    const date = new Date(item.started_at);
    const heading = element('h3',date.toLocaleTimeString('id-ID',{hour12:false}));
    heading.append(element('span',labels[item.status] || item.status,'badge'));
    details.append(heading,element('p',`${date.toLocaleDateString('id-ID')} · ${item.duration.toFixed(1)} detik · ${item.device}`));
    if (item.vad_error) {
      details.append(element('p','Pemeriksaan otomatis gagal. Audio tetap tersimpan dan dapat diputar.','error-detail'));
      const error = document.createElement('details');
      error.append(element('summary','Detail kesalahan'),element('p',item.vad_error,'error-detail'));
      details.append(error);
    }
    const audio = document.createElement('audio'); audio.controls = true; audio.preload = 'none'; audio.src = item.audio_url;
    audio.setAttribute('aria-label',`Putar rekaman ${date.toLocaleTimeString('id-ID')}`);
    audio.addEventListener('play',() => document.querySelectorAll('audio').forEach(a => {if(a !== audio) a.pause();}));
    row.append(details,audio); list.append(row);
  }
}
function recordingsQuery() {
  const query = new URLSearchParams({status:filter, page:String(page), page_size:'20'});
  if (selectedDate) query.set('date',selectedDate);
  return query.toString();
}
function changeRecordingsPage(nextPage = 1) {
  page = nextPage;
  listSignature = '';
  listLoading = true;
  $('previous-page').disabled = $('next-page').disabled = true;
  $('pagination-summary').textContent = 'Memuat rekaman...';
  $('page-status').textContent = '';
  document.querySelectorAll('audio').forEach(audio => audio.pause());
}
async function command(path, body) {
  busy = true; controls(); notice();
  try {
    renderStatus(await api(path,{method:'POST', headers:{'Content-Type':'application/json','X-Recorder-Control':'1'},body:JSON.stringify(body || {})}));
  } catch (error) { notice(error.message); }
  finally { busy = false; controls(); }
}
async function poll() {
  try {
    const status = await api('/api/status');
    if (!connected) notice();
    connected = true; renderStatus(status);
    const query = recordingsQuery();
    const recordings = await api(`/api/recordings?${query}`);
    if (query === recordingsQuery()) renderRecordings(recordings);
  } catch (error) {
    waveform.update(waveform.data, false);
    $('previous-page').disabled = $('next-page').disabled = true;
    connected = false; controls(); notice(`Koneksi ke aplikasi terputus. Pastikan server lokal masih berjalan. ${error.message}`);
  } finally { setTimeout(poll,500); }
}
$('start').addEventListener('click',() => command('/api/start',{
  device_id:Number($('device').value),
  channels:$('channels').value === 'auto' ? null : Number($('channels').value)
}));
$('stop').addEventListener('click',() => command('/api/stop'));
$('refresh').addEventListener('click',loadDevices);
$('device').addEventListener('change',() => {
  $('channels').value = 'auto';
  deviceHint();
});
$('channels').addEventListener('change',deviceHint);
$('all-drivers').addEventListener('change',populateDevices);
document.querySelectorAll('.tab').forEach(button => button.addEventListener('click',() => {
  filter = button.dataset.filter;
  document.querySelectorAll('.tab').forEach(b => {
    b.classList.toggle('selected',b === button);
    b.setAttribute('aria-pressed',String(b === button));
  });
  changeRecordingsPage();
}));
$('recording-date').addEventListener('change',() => {
  selectedDate = $('recording-date').value;
  $('clear-date').disabled = !selectedDate;
  changeRecordingsPage();
});
$('clear-date').addEventListener('click',() => {
  selectedDate = $('recording-date').value = '';
  $('clear-date').disabled = true;
  changeRecordingsPage();
});
$('previous-page').addEventListener('click',() => {
  if (!listLoading && page > 1) changeRecordingsPage(page-1);
});
$('next-page').addEventListener('click',() => {
  if (!listLoading) changeRecordingsPage(page+1);
});
function clock() {
  const now = new Date();
  $('clock').textContent = now.toLocaleTimeString('id-ID',{hour12:false});
  $('date').textContent = now.toLocaleDateString('id-ID',{day:'numeric',month:'short',year:'numeric'});
}
clock(); setInterval(clock,1000); loadDevices(); poll();
