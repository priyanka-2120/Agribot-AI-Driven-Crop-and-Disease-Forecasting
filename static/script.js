function changeLanguage() {
    const language = document.getElementById('language').value;
    const formData = new FormData();
    formData.append('language', language);

    fetch('/set_language', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        const chatBox = document.getElementById('chatBox');
        chatBox.innerHTML = '';
        const botMessage = document.createElement('div');
        botMessage.className = 'message bot-message';
        botMessage.innerHTML = `<p>${data.response}</p>`;
        chatBox.appendChild(botMessage);
        chatBox.scrollTop = chatBox.scrollHeight;
        document.getElementById('imageUpload').style.display = 'none'; // Hide image upload on language change
    });
}

function sendMessage() {
    const userInput = document.getElementById('userInput').value.trim();
    if (userInput === '') return;

    const chatBox = document.getElementById('chatBox');
    const userMessage = document.createElement('div');
    userMessage.className = 'message user-message';
    userMessage.innerHTML = `<p>${userInput}</p>`;
    chatBox.appendChild(userMessage);

    document.getElementById('userInput').value = '';

    const formData = new FormData();
    formData.append('message', userInput);

    fetch('/chat', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        const botMessage = document.createElement('div');
        botMessage.className = 'message bot-message';
        botMessage.innerHTML = `<p>${data.response}</p>`;

        if (data.crop_image) {
            const cropImage = document.createElement('img');
            cropImage.src = data.crop_image;
            cropImage.className = 'crop-image';
            botMessage.appendChild(cropImage);
        }

        chatBox.appendChild(botMessage);

        if (data.request_image) {
            document.getElementById('imageUpload').style.display = 'block';
        } else {
            document.getElementById('imageUpload').style.display = 'none';
        }

        chatBox.scrollTop = chatBox.scrollHeight;
    })
    .catch(error => console.error('Error:', error));
}

function uploadImage() {
    const imageInput = document.getElementById('imageInput');
    if (imageInput.files.length === 0) return;

    const formData = new FormData();
    formData.append('image', imageInput.files[0]);

    fetch('/chat', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        const chatBox = document.getElementById('chatBox');
        const botMessage = document.createElement('div');
        botMessage.className = 'message bot-message';
        botMessage.innerHTML = `<p>${data.response}</p>`;
        chatBox.appendChild(botMessage);

        document.getElementById('imageUpload').style.display = 'none';
        imageInput.value = '';

        chatBox.scrollTop = chatBox.scrollHeight;
    })
    .catch(error => console.error('Error:', error));
}

function downloadChat() {
    fetch('/download_chat', {
        method: 'GET'
    })
    .then(response => {
        if (!response.ok) throw new Error('Failed to download chat history');
        return response.blob();
    })
    .then(blob => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'agribot_chat.html';
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
    })
    .catch(error => {
        console.error('Error downloading chat history:', error);
        alert('Failed to download chat history. Please try again.');
    });
}

// Initial load to ensure welcome message is displayed
document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chatBox');
    const initialMessage = chatBox.querySelector('.bot-message p').textContent;
    chatBox.innerHTML = ''; // Clear the static welcome message
    addMessage(initialMessage, true);
    document.getElementById('imageUpload').style.display = 'none'; // Ensure image upload is hidden initially

    function addMessage(message, isBot, imageUrl = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isBot ? 'bot-message' : 'user-message'}`;
        const p = document.createElement('p');
        p.textContent = message;
        messageDiv.appendChild(p);
        if (imageUrl) {
            const img = document.createElement('img');
            img.src = imageUrl;
            img.className = 'crop-image';
            messageDiv.appendChild(img);
        }
        chatBox.appendChild(messageDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
    }
});
