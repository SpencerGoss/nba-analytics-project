const CACHE = 'nba-analytics-v4';
const STATIC_ASSETS = ['./', './about.html', './manifest.json'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(STATIC_ASSETS)));
  self.skipWaiting();
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  const isData = url.pathname.endsWith('.json') && url.pathname.includes('/data/');
  const isHTML = url.pathname.endsWith('.html') || url.pathname.endsWith('/');

  if (isData || isHTML) {
    // Network-first for JSON data AND HTML — always serve fresh content
    e.respondWith(
      fetch(e.request)
        .then(resp => {
          const clone = resp.clone();
          caches.open(CACHE).then(c => c.put(e.request, clone));
          return resp;
        })
        .catch(() => caches.match(e.request))
    );
  } else {
    // Cache-first for static assets (fonts, images)
    e.respondWith(
      caches.match(e.request).then(cached => {
        if (cached) return cached;
        return fetch(e.request).then(resp => {
          if (resp.ok && url.origin === self.location.origin) {
            const clone = resp.clone();
            caches.open(CACHE).then(c => c.put(e.request, clone));
          }
          return resp;
        });
      })
    );
  }
});
