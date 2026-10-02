/* Run explicitly against the production URL. --jobs makes one paid generation. */
const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const base = process.env.METASYMBO_TEST_URL;
assert(base, 'Set METASYMBO_TEST_URL to the actual endpoint');
const artifacts = path.resolve(__dirname, '../deploy/.runtime');
const reopen = process.argv.includes('--reopen');
const jobs = process.argv.includes('--jobs');

(async () => {
  const browser = await chromium.launch({headless:true, args:['--no-sandbox','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader']});
  try {
    const context = await browser.newContext({viewport:{width:1440,height:1100}, ...(reopen?{storageState:path.join(artifacts,'browser-state.json')}:{})});
    const page = await context.newPage();
    const errors=[];
    page.on('pageerror', error=>errors.push(error.message));
    for(let attempt=0;attempt<30;attempt++) {
      try {if((await page.request.get(base+'/api/health')).ok())break;}catch{}
      await page.waitForTimeout(1000);
    }
    await page.goto(base);
    await page.waitForFunction(()=>document.querySelector('#lattice-view')?.data?.length>0, null, {timeout:60000});
    assert.equal(await page.locator('#session-title').innerText(), 'Try MetaSymbO Online');
    console.log('PASS landing, static assets, WebGL lattice, candidate loading');

    async function openRun(id) {
      const response=await page.request.get(base+'/api/runs/'+id);
      assert.equal(response.status(),200);
      const run=await response.json();
      assert.equal(run.state,'complete',run.stage);
      await page.evaluate(id=>selectCandidate(id),run.selected_candidate);
      await page.waitForFunction(()=>document.querySelector('#lattice-view').data?.length>0);
      assert.equal(await page.locator('#evidence-badge').innerText(),'Predicted');
      const candidate=await (await page.request.get(base+'/api/candidates/'+run.selected_candidate)).json();
      assert.equal(candidate.predictions.length,12);
      if(run.kind==='generation')assert.equal(typeof candidate.provenance.supervisor_score,'number');
      const exported=await page.request.get(base+'/api/candidates/'+run.selected_candidate+'/download');
      assert.equal(exported.status(),200);
      assert.equal((await exported.body()).subarray(0,2).toString(),'PK');
      return candidate;
    }

    if(reopen) {
      for(const id of JSON.parse(fs.readFileSync(path.join(artifacts,'smoke-runs.json'))))await openRun(id);
      console.log('PASS service restart preserves sessions, finished results, predictions, supervisor evidence, and exports');
    } else {
      await page.locator('#compare-add').click();
      await page.locator('nav [data-page="library"]').click();
      await page.waitForSelector('.library-card');
      assert.equal(await page.locator('.library-card').count(),12);
      await page.locator('.library-card input[data-compare]').nth(1).check();
      await page.locator('nav [data-page="compare"]').click();
      await page.waitForFunction(()=>document.querySelectorAll('.compare-view').length===2&&[...document.querySelectorAll('.compare-view')].every(x=>x.data));
      await page.locator('nav [data-page="design"]').click();
      await page.locator('[data-camera="xy"]').click();
      const initialNodes=await page.evaluate(()=>document.querySelector('#lattice-view').data[1].x.length);
      await page.locator('#repeat').selectOption('2');
      await page.waitForFunction(n=>document.querySelector('#lattice-view').data[1].x.length===n*8,initialNodes);
      await page.locator('#repeat').selectOption('1');
      console.log('PASS comparison, cell repetition, camera controls');

      const source=(await (await page.request.get(base+'/api/candidates?limit=1')).json()).items[0];
      const bytes=await (await page.request.get(base+'/api/candidates/'+source.id+'/download')).body();
      const imported=await page.request.post(base+'/api/import',{data:bytes,headers:{'Content-Type':'application/octet-stream'}});
      assert.equal(imported.status(),201);
      assert.equal((await imported.json()).fingerprint,source.fingerprint);
      assert.equal((await page.request.post(base+'/api/import',{data:'invalid'})).status(),422);
      assert.equal((await page.request.post(base+'/api/runs',{data:{kind:'generation',prompt:'x'.repeat(2001)}})).status(),422);
      assert.equal((await page.request.post(base+'/api/runs',{data:{kind:'generation',prompt:'x',attempts:3}})).status(),422);
      assert.equal((await page.request.post(base+'/api/runs',{headers:{Origin:'https://attacker.invalid'},data:{kind:'generation',prompt:'x'}})).status(),403);
      for(const url of ['/.env','/WebInterface/.env','/../.env','/etc/passwd','/static/../.env','/static/%2e%2e/.env','/api/runs/invalid','/checkpoints/best_ae_model.pt']) {
        assert.equal((await page.request.get(base+url)).status(),404,url);
      }
      console.log('PASS import/export, malformed requests, private paths, traversal, cross-origin writes');

      if(jobs) {
        const ids=[];
        for(const payload of [{kind:'prediction',candidate_id:source.id,seed:42}, {kind:'generation',prompt:'Design a simple three-dimensional connected truss lattice with fewer than 20 nodes and high stiffness along Z. Use a straightforward scaffold.',attempts:1,seed:42,logic_mode:'union'}]) {
          const response=await page.request.post(base+'/api/runs',{data:payload});
          assert.equal(response.status(),202,await response.text());
          const id=(await response.json()).id;
          ids.push(id);
          const duplicate=await page.request.post(base+'/api/runs',{data:payload});
          assert.equal(duplicate.status(),409);
          assert.match((await duplicate.json()).detail,/currently busy/);
          console.log('Started '+payload.kind+' '+id+'; duplicate request rejected');
          let previous='';
          const deadline=Date.now()+930000;
          while(Date.now()<deadline) {
            const run=await (await page.request.get(base+'/api/runs/'+id)).json();
            if(run.stage!==previous){console.log(payload.kind+': '+run.stage);previous=run.stage;}
            if(['complete','failed','interrupted'].includes(run.state)){assert.equal(run.state,'complete',run.stage);break;}
            await page.waitForTimeout(5000);
          }
          await openRun(id);
          console.log('PASS live '+payload.kind+' and saved result');
        }
        fs.writeFileSync(path.join(artifacts,'smoke-runs.json'),JSON.stringify(ids),{mode:0o600});
        await context.storageState({path:path.join(artifacts,'browser-state.json')});
        fs.chmodSync(path.join(artifacts,'browser-state.json'),0o600);
      }
      await page.locator('#brief').fill('A simple truss with stiffness along Z.');
      await page.locator('#review-button').click();
      await page.locator('#review-dialog').waitFor({state:'visible'});
      assert.match(await page.locator('#review-text').innerText(),/simple truss/);
      await page.locator('#review-edit').click();
      await page.reload();
      await page.waitForFunction(()=>document.querySelector('#lattice-view').data?.length>0);
      assert.match(await page.locator('#brief').inputValue(),/simple truss/);
    }
    await page.locator('#tab-properties').click();
    if(jobs||reopen)await page.waitForSelector('.property-chart');
    await page.locator('#tab-structure').click();
    await page.screenshot({path:path.join(artifacts,'production-desktop.png'),fullPage:true});
    await page.setViewportSize({width:390,height:844});
    await page.waitForTimeout(1000);
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,'Mobile overflow');
    await page.screenshot({path:path.join(artifacts,'production-mobile.png'),fullPage:true});
    assert.deepEqual(errors,[]);
    console.log('PASS refresh persistence, mobile layout, zero browser errors');
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exit(1);});
