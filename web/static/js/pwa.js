/**
 * RDT Trading System - PWA JavaScript Module
 *
 * Provides:
 * - Service worker registration
 * - Install prompt handling
 * - Online/offline status detection
 * - Push notification subscription
 * - Background sync registration
 */

(function() {
  'use strict';

  // PWA Configuration
  const PWA_CONFIG = {
    serviceWorkerPath: '/sw.js',
    vapidPublicKey: null, // Set from server configuration
    notificationEndpoint: '/api/push/subscribe',
    debug: false
  };

  // PWA State
  const PWAState = {
    serviceWorkerRegistration: null,
    installPromptEvent: null,
    isOnline: navigator.onLine,
    isInstalled: false,
    notificationPermission: Notification.permission,
    pushSubscription: null
  };

  // Debug logging
  function log(...args) {
    if (PWA_CONFIG.debug) {
      console.log('[PWA]', ...args);
    }
  }

  // =========================================================================
  // Service Worker Registration
  // =========================================================================

  async function registerServiceWorker() {
    if (!('serviceWorker' in navigator)) {
      log('Service workers not supported');
      return null;
    }

    // Unregister any existing service workers to prevent fetch interception issues
    try {
      const registrations = await navigator.serviceWorker.getRegistrations();
      for (const registration of registrations) {
        await registration.unregister();
        log('Unregistered service worker:', registration.scope);
      }
      if (registrations.length > 0) {
        log('Cleared', registrations.length, 'service worker(s)');
      }
    } catch (error) {
      log('Error clearing service workers:', error);
    }

    return null;
  }

  // Show update notification to user
  function showUpdateNotification() {
    const updateBanner = document.createElement('div');
    updateBanner.id = 'pwa-update-banner';
    updateBanner.innerHTML = `
      <div class="pwa-update-content">
        <span>A new version is available!</span>
        <button onclick="window.PWA.applyUpdate()">Update Now</button>
        <button onclick="this.parentElement.parentElement.remove()">Later</button>
      </div>
    `;
    updateBanner.style.cssText = `
      position: fixed;
      bottom: 20px;
      left: 50%;
      transform: translateX(-50%);
      background: var(--primary, #1a1a2e);
      color: white;
      padding: 15px 25px;
      border-radius: 8px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
      z-index: 10000;
      display: flex;
      align-items: center;
      gap: 15px;
    `;

    const style = document.createElement('style');
    style.textContent = `
      #pwa-update-banner button {
        background: var(--accent, #4CAF50);
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 4px;
        cursor: pointer;
        font-size: 14px;
      }
      #pwa-update-banner button:hover {
        opacity: 0.9;
      }
      #pwa-update-banner button:last-child {
        background: transparent;
        border: 1px solid rgba(255,255,255,0.3);
      }
    `;
    document.head.appendChild(style);
    document.body.appendChild(updateBanner);
  }

  // Apply service worker update
  function applyUpdate() {
    if (PWAState.serviceWorkerRegistration && PWAState.serviceWorkerRegistration.waiting) {
      PWAState.serviceWorkerRegistration.waiting.postMessage({ type: 'SKIP_WAITING' });
    }
    window.location.reload();
  }

  // =========================================================================
  // Install Prompt Handling
  // =========================================================================

  function setupInstallPrompt() {
    // Check if already installed
    if (window.matchMedia('(display-mode: standalone)').matches) {
      PWAState.isInstalled = true;
      log('App is already installed');
      return;
    }

    // Listen for beforeinstallprompt event
    window.addEventListener('beforeinstallprompt', (event) => {
      event.preventDefault();
      PWAState.installPromptEvent = event;
      log('Install prompt captured');

      // Show install button
      showInstallButton();

      // Dispatch custom event
      window.dispatchEvent(new CustomEvent('pwa:installable'));
    });

    // Track installation
    window.addEventListener('appinstalled', () => {
      PWAState.isInstalled = true;
      PWAState.installPromptEvent = null;
      log('App installed');

      hideInstallButton();

      // Dispatch custom event
      window.dispatchEvent(new CustomEvent('pwa:installed'));
    });
  }

  // Show install button in UI
  function showInstallButton() {
    // Check if button already exists
    if (document.getElementById('pwa-install-button')) {
      document.getElementById('pwa-install-button').style.display = 'flex';
      return;
    }

    const installButton = document.createElement('button');
    installButton.id = 'pwa-install-button';
    installButton.innerHTML = `
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
        <polyline points="7 10 12 15 17 10"/>
        <line x1="12" y1="15" x2="12" y2="3"/>
      </svg>
      <span>Install App</span>
    `;
    installButton.style.cssText = `
      position: fixed;
      bottom: 20px;
      right: 20px;
      background: var(--accent, #4CAF50);
      color: white;
      border: none;
      padding: 12px 20px;
      border-radius: 25px;
      cursor: pointer;
      font-size: 14px;
      font-weight: 500;
      display: flex;
      align-items: center;
      gap: 8px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.2);
      z-index: 9999;
      transition: transform 0.2s, box-shadow 0.2s;
    `;

    installButton.addEventListener('mouseenter', () => {
      installButton.style.transform = 'scale(1.05)';
      installButton.style.boxShadow = '0 6px 16px rgba(0,0,0,0.3)';
    });

    installButton.addEventListener('mouseleave', () => {
      installButton.style.transform = 'scale(1)';
      installButton.style.boxShadow = '0 4px 12px rgba(0,0,0,0.2)';
    });

    installButton.addEventListener('click', promptInstall);
    document.body.appendChild(installButton);
  }

  // Hide install button
  function hideInstallButton() {
    const button = document.getElementById('pwa-install-button');
    if (button) {
      button.style.display = 'none';
    }
  }

  // Prompt user to install
  async function promptInstall() {
    if (!PWAState.installPromptEvent) {
      log('No install prompt available');
      return false;
    }

    PWAState.installPromptEvent.prompt();
    const result = await PWAState.installPromptEvent.userChoice;
    log('Install prompt result:', result.outcome);

    PWAState.installPromptEvent = null;

    if (result.outcome === 'accepted') {
      hideInstallButton();
      return true;
    }

    return false;
  }

  // =========================================================================
  // Online/Offline Status Detection
  // =========================================================================

  function setupOnlineStatus() {
    // Create offline indicator element
    const offlineIndicator = document.createElement('div');
    offlineIndicator.id = 'pwa-offline-indicator';
    offlineIndicator.innerHTML = `
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <line x1="1" y1="1" x2="23" y2="23"/>
        <path d="M16.72 11.06A10.94 10.94 0 0 1 19 12.55"/>
        <path d="M5 12.55a10.94 10.94 0 0 1 5.17-2.39"/>
        <path d="M10.71 5.05A16 16 0 0 1 22.58 9"/>
        <path d="M1.42 9a15.91 15.91 0 0 1 4.7-2.88"/>
        <path d="M8.53 16.11a6 6 0 0 1 6.95 0"/>
        <line x1="12" y1="20" x2="12.01" y2="20"/>
      </svg>
      <span>You are offline</span>
    `;
    offlineIndicator.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      background: var(--warning, #f39c12);
      color: white;
      padding: 10px;
      text-align: center;
      display: none;
      align-items: center;
      justify-content: center;
      gap: 8px;
      font-size: 14px;
      z-index: 10001;
    `;
    document.body.appendChild(offlineIndicator);

    // Update status
    function updateOnlineStatus() {
      PWAState.isOnline = navigator.onLine;
      offlineIndicator.style.display = PWAState.isOnline ? 'none' : 'flex';

      log('Online status:', PWAState.isOnline);

      // Dispatch custom event
      window.dispatchEvent(new CustomEvent('pwa:onlinechange', {
        detail: { isOnline: PWAState.isOnline }
      }));

      // If back online, trigger sync
      if (PWAState.isOnline && PWAState.serviceWorkerRegistration) {
        triggerBackgroundSync('sync-pending-actions');
      }
    }

    window.addEventListener('online', updateOnlineStatus);
    window.addEventListener('offline', updateOnlineStatus);

    // Initial check
    updateOnlineStatus();
  }

  // =========================================================================
  // Push Notification Subscription
  // =========================================================================

  async function requestNotificationPermission() {
    if (!('Notification' in window)) {
      log('Notifications not supported');
      return false;
    }

    if (Notification.permission === 'granted') {
      PWAState.notificationPermission = 'granted';
      return true;
    }

    if (Notification.permission === 'denied') {
      PWAState.notificationPermission = 'denied';
      log('Notification permission denied');
      return false;
    }

    const permission = await Notification.requestPermission();
    PWAState.notificationPermission = permission;
    log('Notification permission:', permission);

    return permission === 'granted';
  }

  async function subscribeToPushNotifications() {
    if (!PWAState.serviceWorkerRegistration) {
      log('Service worker not registered');
      return null;
    }

    const hasPermission = await requestNotificationPermission();
    if (!hasPermission) {
      return null;
    }

    try {
      // Get VAPID public key from server
      const vapidKey = await getVapidPublicKey();
      if (!vapidKey) {
        log('VAPID public key not available');
        return null;
      }

      const subscription = await PWAState.serviceWorkerRegistration.pushManager.subscribe({
        userVisibleOnly: true,
        applicationServerKey: urlBase64ToUint8Array(vapidKey)
      });

      PWAState.pushSubscription = subscription;
      log('Push subscription:', subscription);

      // Send subscription to server
      await sendSubscriptionToServer(subscription);

      return subscription;
    } catch (error) {
      console.error('[PWA] Push subscription failed:', error);
      return null;
    }
  }

  async function unsubscribeFromPushNotifications() {
    if (!PWAState.pushSubscription) {
      return true;
    }

    try {
      await PWAState.pushSubscription.unsubscribe();

      // Notify server
      await fetch('/api/push/unsubscribe', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': window.getCSRFToken ? window.getCSRFToken() : ''
        },
        body: JSON.stringify({
          endpoint: PWAState.pushSubscription.endpoint
        })
      });

      PWAState.pushSubscription = null;
      log('Unsubscribed from push notifications');

      return true;
    } catch (error) {
      console.error('[PWA] Unsubscribe failed:', error);
      return false;
    }
  }

  async function getVapidPublicKey() {
    if (PWA_CONFIG.vapidPublicKey) {
      return PWA_CONFIG.vapidPublicKey;
    }

    try {
      const response = await fetch('/api/push/vapid-key');
      const data = await response.json();
      PWA_CONFIG.vapidPublicKey = data.publicKey;
      return data.publicKey;
    } catch (error) {
      console.error('[PWA] Failed to get VAPID key:', error);
      return null;
    }
  }

  async function sendSubscriptionToServer(subscription) {
    try {
      const response = await fetch(PWA_CONFIG.notificationEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': window.getCSRFToken ? window.getCSRFToken() : ''
        },
        body: JSON.stringify(subscription)
      });

      if (!response.ok) {
        throw new Error('Server rejected subscription');
      }

      log('Subscription sent to server');
      return true;
    } catch (error) {
      console.error('[PWA] Failed to send subscription:', error);
      return false;
    }
  }

  // Convert VAPID key
  function urlBase64ToUint8Array(base64String) {
    const padding = '='.repeat((4 - base64String.length % 4) % 4);
    const base64 = (base64String + padding)
      .replace(/-/g, '+')
      .replace(/_/g, '/');

    const rawData = window.atob(base64);
    const outputArray = new Uint8Array(rawData.length);

    for (let i = 0; i < rawData.length; ++i) {
      outputArray[i] = rawData.charCodeAt(i);
    }
    return outputArray;
  }

  // =========================================================================
  // Background Sync
  // =========================================================================

  async function triggerBackgroundSync(tag) {
    if (!PWAState.serviceWorkerRegistration) {
      return false;
    }

    if (!('sync' in PWAState.serviceWorkerRegistration)) {
      log('Background sync not supported');
      return false;
    }

    try {
      await PWAState.serviceWorkerRegistration.sync.register(tag);
      log('Background sync registered:', tag);
      return true;
    } catch (error) {
      console.error('[PWA] Background sync registration failed:', error);
      return false;
    }
  }

  // Queue action for background sync
  async function queueActionForSync(action) {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open('rdt-trading-sw', 1);

      request.onerror = () => reject(request.error);

      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        if (!db.objectStoreNames.contains('pending-actions')) {
          db.createObjectStore('pending-actions', { keyPath: 'id', autoIncrement: true });
        }
      };

      request.onsuccess = () => {
        const db = request.result;
        const tx = db.transaction('pending-actions', 'readwrite');
        const store = tx.objectStore('pending-actions');

        store.add({
          ...action,
          timestamp: Date.now()
        });

        tx.oncomplete = () => {
          triggerBackgroundSync('sync-pending-actions');
          resolve(true);
        };

        tx.onerror = () => reject(tx.error);
      };
    });
  }

  // =========================================================================
  // Cache Management
  // =========================================================================

  async function getCacheStatus() {
    if (!PWAState.serviceWorkerRegistration) {
      return null;
    }

    return new Promise((resolve) => {
      const messageChannel = new MessageChannel();

      messageChannel.port1.onmessage = (event) => {
        resolve(event.data);
      };

      navigator.serviceWorker.controller.postMessage(
        { type: 'GET_CACHE_STATUS' },
        [messageChannel.port2]
      );
    });
  }

  async function clearCache() {
    if (!navigator.serviceWorker.controller) {
      return false;
    }

    navigator.serviceWorker.controller.postMessage({ type: 'CLEAR_CACHE' });
    log('Cache cleared');
    return true;
  }

  // =========================================================================
  // Initialization
  // =========================================================================

  async function init() {
    log('Initializing PWA...');

    // Register service worker
    await registerServiceWorker();

    // Setup install prompt
    setupInstallPrompt();

    // Setup online/offline detection
    setupOnlineStatus();

    // Check existing push subscription
    if (PWAState.serviceWorkerRegistration) {
      try {
        PWAState.pushSubscription = await PWAState.serviceWorkerRegistration.pushManager.getSubscription();
        if (PWAState.pushSubscription) {
          log('Existing push subscription found');
        }
      } catch (error) {
        log('Error checking push subscription:', error);
      }
    }

    log('PWA initialized');

    // Dispatch ready event
    window.dispatchEvent(new CustomEvent('pwa:ready', {
      detail: { state: PWAState }
    }));
  }

  // =========================================================================
  // Public API
  // =========================================================================

  window.PWA = {
    // State
    get state() { return { ...PWAState }; },
    get isOnline() { return PWAState.isOnline; },
    get isInstalled() { return PWAState.isInstalled; },
    get canInstall() { return !!PWAState.installPromptEvent; },
    get notificationPermission() { return PWAState.notificationPermission; },

    // Methods
    init,
    promptInstall,
    applyUpdate,
    requestNotificationPermission,
    subscribeToPushNotifications,
    unsubscribeFromPushNotifications,
    triggerBackgroundSync,
    queueActionForSync,
    getCacheStatus,
    clearCache,

    // Configuration
    setDebug(enabled) { PWA_CONFIG.debug = enabled; },
    setVapidKey(key) { PWA_CONFIG.vapidPublicKey = key; }
  };

  // Auto-initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
