const CACHE = 'baseline-v1';
const ASSETS = ['./','./index.html','./data/todays_picks.json','./data/value_bets.json','./data/meta.json'];
self.addEventListener('install', e => e.waitUntil(caches.open(CACHE).then(c => c.addAll(ASSETS))));
self.addEventListener('fetch', e => e.respondWith(
  caches.match(e.request).then(r => r || fetch(e.request))
));
