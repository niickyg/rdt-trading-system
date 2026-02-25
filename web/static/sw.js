/**
 * RDT Trading System - Service Worker (DISABLED)
 *
 * This service worker immediately unregisters itself and clears all caches.
 * The PWA service worker was causing fetch interception issues with POST requests.
 */

// Self-destruct: unregister this service worker and clear all caches
self.addEventListener('install', () => {
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => caches.delete(cacheName))
      );
    }).then(() => {
      return self.registration.unregister();
    }).then(() => {
      return self.clients.matchAll();
    }).then((clients) => {
      clients.forEach((client) => client.navigate(client.url));
    })
  );
});
