import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

// Exercise the page's actual event handlers without audio hardware or a browser.
function setup() {
  const nodes = new Map();
  class Node {
    constructor() { this.children = []; this.listeners = {}; this.value = ''; }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; }
    addEventListener(event, listener) { this.listeners[event] = listener; }
    setAttribute() {}
    pause() {}
  }
  const get = id => {
    if (!nodes.has(id)) nodes.set(id, new Node());
    return nodes.get(id);
  };
  const sandbox = vm.createContext({
    document: {
      getElementById: get,
      querySelectorAll: () => [],
      createElement: () => Object.assign(new Node(), { dataset: {} }),
    },
    WaveformView: class {},
    URLSearchParams,
    window: { location: { assign(url) { sandbox.download = url; } } },
  });
  const source = readFileSync(new URL('../web/app.js', import.meta.url), 'utf8')
    .replace(/^import .*;\r?\n/gm, '')
    .replace('clock(); setInterval(clock,1000); loadDevices(); poll();', '');
  vm.runInContext(source, sandbox);
  const run = code => vm.runInContext(code, sandbox);
  const render = (page, id) => {
    run(`renderRecordings({counts:{speech:30}, total:30, page:${page}, page_size:20,
      total_pages:2, items:[{id:'${id}',started_at:'2026-10-23T09:00:00+07:00',
      status:'speech',duration:1,device:'USB',audio_url:'/audio'}]})`);
    return get('recording-list').children[0].children[0];
  };
  return { get, run, render, sandbox };
}

test('individual choices survive pagination and produce explicit download IDs', () => {
  const { get, run, render, sandbox } = setup();
  const first = render(1, 'first');
  first.checked = true;
  first.listeners.change();
  run('changeRecordingsPage(2)');
  const last = render(2, 'last');
  last.checked = true;
  last.listeners.change();
  assert.equal(get('download-recordings').textContent, 'Download terpilih (2)');
  get('download-recordings').listeners.click();
  assert.deepEqual(new URLSearchParams(sandbox.download.split('?')[1]).getAll('id'), ['first','last']);
  run('changeRecordingsPage(1)');
  assert.equal(render(1, 'first').checked, true);
});

test('select all includes unseen pages, supports exclusions, and resets on date change', () => {
  const { get, run, render, sandbox } = setup();
  render(1, 'first');
  get('select-all-recordings').checked = true;
  get('select-all-recordings').listeners.change();
  assert.equal(get('download-recordings').textContent, 'Download terpilih (30)');
  run('changeRecordingsPage(2)');
  const last = render(2, 'last');
  assert.equal(last.checked, true);
  last.checked = false;
  last.listeners.change();
  assert.equal(get('download-recordings').textContent, 'Download terpilih (29)');
  get('download-recordings').listeners.click();
  assert.equal(sandbox.download, '/api/recordings/download?status=speech&exclude=last');
  get('recording-date').value = '2026-10-24';
  get('recording-date').listeners.change();
  assert.equal(get('download-recordings').textContent, 'Download terpilih (0)');
  assert.equal(get('download-recordings').disabled, true);
});

test('trash requires confirmation and sends the cross-page selection with its count', async () => {
  const { get, render, sandbox } = setup();
  render(1, 'first');
  get('select-all-recordings').checked = true;
  get('select-all-recordings').listeners.change();
  const calls = [];
  sandbox.fetch = async (url, options) => {
    calls.push({url, body:JSON.parse(options.body), headers:options.headers});
    return {ok:true, json:async () => ({count:30})};
  };
  sandbox.window.confirm = () => false;
  await get('trash-recordings').listeners.click();
  assert.equal(calls.length, 0);
  assert.equal(get('download-recordings').textContent, 'Download terpilih (30)');
  sandbox.window.confirm = message => {
    assert.match(message, /Pindahkan ke Sampah 30 rekaman/);
    assert.match(message, /semua halaman/);
    return true;
  };
  await get('trash-recordings').listeners.click();
  assert.equal(calls[0].url, '/api/recordings/trash');
  assert.equal(calls[0].body.expected_count, 30);
  assert.equal(calls[0].body.select_all, true);
  assert.equal(calls[0].headers['X-Recorder-Control'], '1');
  assert.equal(get('download-recordings').textContent, 'Download terpilih (0)');
});

test('Trash view offers restore and preserves selection on server failure', async () => {
  const { get, run, render, sandbox } = setup();
  run("filter = 'trash'");
  const checkbox = render(1, 'trashed');
  checkbox.checked = true;
  checkbox.listeners.change();
  assert.equal(get('trash-recordings').hidden, true);
  assert.equal(get('restore-recordings').hidden, false);
  sandbox.window.confirm = () => true;
  sandbox.fetch = async url => {
    assert.equal(url, '/api/recordings/restore');
    return {ok:false, json:async () => ({detail:'Daftar berubah'})};
  };
  await get('restore-recordings').listeners.click();
  assert.equal(get('notice').textContent, 'Daftar berubah');
  assert.equal(get('restore-recordings').textContent, 'Pulihkan terpilih (1)');
  assert.equal(get('restore-recordings').disabled, false);
});
