import assert from 'node:assert/strict';
import test from 'node:test';
import {peakHeight, timeLabel, WaveformView} from '../web/waveform.mjs';

test('tinggi gelombang mempertahankan hening dan waktu terbaca sampai sesi panjang', () => {
  assert.equal(peakHeight(0,90),0);
  assert.equal(peakHeight(.25,90),45);
  assert.equal(peakHeight(1,90),90);
  assert.equal(peakHeight(2,90),90);
  assert.equal(peakHeight(-1,90),0);
  assert.equal(timeLabel(0),'00:00');
  assert.equal(timeLabel(65.8),'01:05');
  assert.equal(timeLabel(3661),'1:01:01');
});

test('canvas memakai sampel backend, mempertahankan puncak negatif, dan tahan resize', () => {
  const strokes = [];
  let path = [];
  const context = new Proxy({}, {get: (target,key) => target[key] || (() => {}),
    set: (target,key,value) => {target[key]=value; return true;}});
  context.beginPath = () => {path = [];};
  context.moveTo = (...args) => path.push(['move',...args]);
  context.lineTo = (...args) => path.push(['line',...args]);
  context.stroke = () => strokes.push({color:context.strokeStyle,path:structuredClone(path)});
  globalThis.window = {devicePixelRatio:2, matchMedia:() => ({matches:true})};
  globalThis.ResizeObserver = class {observe() {}};
  const canvas = {width:0, height:0, clientWidth:500, clientHeight:174,
    getContext:() => context};
  const view = new WaveformView(canvas);
  const data = {window_seconds:10, elapsed:10,
    points:[{time:-1,min:-1,max:1,rms_db:0},
      {time:.01,min:-.9,max:.9,rms_db:-5},
      {time:9,min:-.2,max:.1,rms_db:-25},
      {time:11,min:-1,max:1,rms_db:0}]};
  view.update(data,false);
  assert.equal(canvas.width,1000);
  assert.equal(canvas.height,348);
  const bars = strokes.find(stroke => stroke.color === '#66747d').path;
  assert.equal(bars.length,4, 'sampel kedaluwarsa dan masa depan tidak digambar');
  assert.ok(bars.every(command => command[1]<250), 'audio hanya di kiri garis merah');
  const halfHeight = (174-14-45)/2;
  assert.ok(Math.abs((bars[3][2]-bars[2][2])-2*peakHeight(.2,halfHeight)) < 1e-9,
    'puncak negatif tetap dihitung meski kanal lain berlawanan fase');
  const cursor = strokes.find(stroke => stroke.color === '#e34c5a').path;
  assert.equal(cursor[0][1],250);
  assert.equal(cursor[1][1],250);
  const snapshot = structuredClone(data);
  canvas.clientWidth = 320;
  view.draw();
  assert.equal(canvas.width,640);
  assert.deepEqual(data,snapshot);
  assert.doesNotThrow(() => view.update(null,false));
  canvas.clientWidth = 0;
  assert.doesNotThrow(() => view.draw());
});
