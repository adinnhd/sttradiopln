// Amplop puncak input Python. Kompresi visual tidak mengubah sampel WAV.
export function peakHeight(amplitude, halfHeight) {
  return Math.sqrt(Math.max(0, Math.min(1, amplitude))) * halfHeight;
}

export function timeLabel(seconds) {
  const total = Math.max(0,Math.floor(seconds));
  const minutes = Math.floor(total/60);
  const label = `${String(minutes%60).padStart(2,'0')}:${String(total%60).padStart(2,'0')}`;
  return minutes >= 60 ? `${Math.floor(minutes/60)}:${label}` : label;
}

export class WaveformView {
  constructor(canvas) {
    this.canvas = canvas;
    this.context = canvas.getContext('2d');
    this.data = {window_seconds:10, elapsed:0, points:[]};
    this.active = false;
    this.receivedAt = performance.now();
    this.reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    this.resize = new ResizeObserver(() => this.draw());
    this.resize.observe(canvas);
    this.frame = 0;
    this.tick = (now) => {
      if (!this.active || this.reducedMotion.matches) {
        this.frame = 0;
        this.draw();
        return;
      }
      if (!this.lastDraw || now - this.lastDraw >= 40) {
        this.draw();
        this.lastDraw = now;
      }
      this.frame = requestAnimationFrame(this.tick);
    };
  }

  update(data, active) {
    this.data = data || {window_seconds:10, elapsed:0, points:[]};
    this.active = active;
    this.receivedAt = performance.now();
    this.draw();
    if (!this.frame && active && !this.reducedMotion.matches) {
      this.frame = requestAnimationFrame(this.tick);
    }
  }

  draw() {
    const ctx = this.context;
    if (!ctx) return;
    const width = this.canvas.clientWidth, height = this.canvas.clientHeight;
    if (!width || !height) return;
    const ratio = window.devicePixelRatio || 1;
    if (this.canvas.width !== Math.round(width*ratio) || this.canvas.height !== Math.round(height*ratio)) {
      this.canvas.width = Math.round(width*ratio);
      this.canvas.height = Math.round(height*ratio);
    }
    ctx.setTransform(ratio,0,0,ratio,0,0);
    ctx.clearRect(0,0,width,height);
    const left = 12, right = width - 12, top = 45, bottom = height - 14;
    const center = (top+bottom)/2, halfHeight = (bottom-top)/2;
    const plotWidth = right-left;
    const cursor = left+plotWidth/2;
    const smoothOffset = this.active && !this.reducedMotion.matches
      ? Math.min(1, (performance.now()-this.receivedAt)/1000) : 0;
    const end = this.data.elapsed + smoothOffset;
    const seconds = this.data.window_seconds;
    const start = end-seconds;
    ctx.font = '12px Arial, sans-serif';
    ctx.lineWidth = 1;
    ctx.strokeStyle = '#e7ebed';
    ctx.beginPath(); ctx.moveTo(left,center); ctx.lineTo(right,center); ctx.stroke();
    ctx.fillStyle = '#6a747c';
    ctx.textAlign = 'center';
    const tickSeconds = width < 400 ? 5 : 4;
    for (let time=Math.ceil(start/tickSeconds)*tickSeconds; time<=end+seconds; time+=tickSeconds) {
      if (time < 0) continue;
      const x = cursor+(time-end)/seconds*(plotWidth/2);
      if (x < left+18 || x > right-18) continue;
      ctx.fillText(timeLabel(time),x,18);
      ctx.strokeStyle = '#bac3c8';
      ctx.beginPath(); ctx.moveTo(x,26); ctx.lineTo(x,32); ctx.stroke();
    }
    ctx.save();
    ctx.beginPath(); ctx.rect(left,top,plotWidth,bottom-top); ctx.clip();
    const bars = new Map();
    for (const point of this.data.points) {
      if (point.time < start || point.time > end) continue;
      const x = Math.min(cursor-2,Math.round((cursor+(point.time-end)/seconds*(plotWidth/2))/4)*4);
      const peak = Math.max(Math.abs(point.min),Math.abs(point.max));
      bars.set(x,Math.max(bars.get(x) || 0,peak));
    }
    ctx.strokeStyle = '#66747d'; ctx.lineWidth = 2.5; ctx.lineCap = 'round';
    ctx.beginPath();
    for (const [x,peak] of bars) {
      const distance = peakHeight(peak,halfHeight);
      if (distance < .3) continue;
      ctx.moveTo(x,center-distance); ctx.lineTo(x,center+distance);
    }
    ctx.stroke();
    ctx.restore();
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#e34c5a';
    ctx.beginPath(); ctx.moveTo(cursor,38); ctx.lineTo(cursor,bottom); ctx.stroke();
  }
}
