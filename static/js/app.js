// Application state
let state = {
    cameraOn: true,
    recognitionOn: true
};

// DOM elements
const elements = {
    cameraBtn: document.getElementById('camera-btn'),
    recognitionBtn: document.getElementById('recognition-btn'),
    cameraStatus: document.getElementById('camera-status'),
    recognitionStatus: document.getElementById('recognition-status')
};

// Initialize application
function init() {
    updateUI();
}

// Toggle camera
async function toggleCamera() {
    try {
        const response = await fetch('/toggle_camera', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        const data = await response.json();
        state.cameraOn = data.camera_on;
        updateUI();
    } catch (error) {
        console.error('Error toggling camera:', error);
    }
}

// Toggle recognition
async function toggleRecognition() {
    try {
        const response = await fetch('/toggle_recognition', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        const data = await response.json();
        state.recognitionOn = data.recognition_on;
        updateUI();
    } catch (error) {
        console.error('Error toggling recognition:', error);
    }
}

// Update UI based on state
function updateUI() {
    // Update camera button
    if (state.cameraOn) {
        elements.cameraBtn.textContent = 'Turn Camera Off';
        elements.cameraBtn.classList.remove('off');
        elements.cameraStatus.textContent = 'Active';
        elements.cameraStatus.classList.add('online');
        elements.cameraStatus.classList.remove('offline');
    } else {
        elements.cameraBtn.textContent = 'Turn Camera On';
        elements.cameraBtn.classList.add('off');
        elements.cameraStatus.textContent = 'Inactive';
        elements.cameraStatus.classList.remove('online');
        elements.cameraStatus.classList.add('offline');
    }

    // Update recognition button
    if (state.recognitionOn) {
        elements.recognitionBtn.textContent = 'Turn Recognition Off';
        elements.recognitionBtn.classList.remove('off');
        elements.recognitionStatus.textContent = 'Active';
        elements.recognitionStatus.classList.add('online');
        elements.recognitionStatus.classList.remove('offline');
    } else {
        elements.recognitionBtn.textContent = 'Turn Recognition On';
        elements.recognitionBtn.classList.add('off');
        elements.recognitionStatus.textContent = 'Inactive';
        elements.recognitionStatus.classList.remove('online');
        elements.recognitionStatus.classList.add('offline');
    }
}

// Event listeners
document.addEventListener('DOMContentLoaded', init);
elements.cameraBtn.addEventListener('click', toggleCamera);
elements.recognitionBtn.addEventListener('click', toggleRecognition);
