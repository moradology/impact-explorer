document.addEventListener('DOMContentLoaded', initializeChat);

const sendButton = document.getElementById('send-btn');
const userInput = document.getElementById('user-input');
const messagesContainer = document.getElementById('messages');
const chatWindow = document.getElementById('chat-window');

sendButton.addEventListener('click', sendMessage);
userInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
        sendMessage();
    }
});

async function initializeChat() {
    try {
        const response = await fetch('/new-context', { method: 'POST' });
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        const data = await response.json();
        
        let formattedMessage = '';
        if (Array.isArray(data.message)) {
            formattedMessage = data.message.map((line, index) => {
                if (index === 0 && data.disclaimer_url) {
                    return line.replace(
                        "terms and conditions",
                        `<a href="${data.disclaimer_url}" target="_blank">terms and conditions</a>`
                    );
                }
                return line;
            }).join('<br>');
        } else {
            formattedMessage = data.message;
        }
        
        addMessage('system', formattedMessage, true);
    } catch (error) {
        console.error('Error:', error);
        addMessage('error', "Error: Unable to start a new chat session.");
    }
    userInput.focus();
}

async function sendMessage() {
    const message = userInput.value.trim();
    if (message === "") return;

    addMessage('user', message);
    userInput.value = '';

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: new URLSearchParams({
                'message': message
            })
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let botMessage = '';
        let botElement = null;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split('\n\n');
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = JSON.parse(line.slice(6));
                    if (data.delta) {
                        if (!botElement) {
                            botElement = document.createElement('div');
                            botElement.className = 'message bot';
                            messagesContainer.appendChild(botElement);
                        }
                        botMessage += data.delta;
                        botElement.textContent = botMessage;
                        scrollToBottom();
                    } else if (data.done) {
                        break;
                    } else if (data.error) {
                        addMessage('error', `Error: ${data.error}`);
                        break;
                    }
                }
            }
        }
    } catch (error) {
        console.error('Error:', error);
        addMessage('error', "Error: Unable to get response from the server.");
    }

    userInput.focus();
}

function addMessage(sender, content, isHTML = false) {
    const messageElement = document.createElement('div');
    messageElement.className = `message ${sender}`;
    if (isHTML) {
        messageElement.innerHTML = content;
    } else {
        messageElement.textContent = content;
    }
    messagesContainer.appendChild(messageElement);
    scrollToBottom();
}

function scrollToBottom() {
    requestAnimationFrame(() => {
        chatWindow.scrollTop = chatWindow.scrollHeight;
    });
}

// Debounce function for smoother scrolling during rapid updates
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Use debounced scroll function for smoother performance
const debouncedScrollToBottom = debounce(scrollToBottom, 100);

// Add event listener for resize to adjust scroll
window.addEventListener('resize', debouncedScrollToBottom);

// Ensure scrolling when the window is resized
window.addEventListener('resize', debouncedScrollToBottom);

// Ensure scrolling when new content is added (for dynamically loaded content)
const observer = new MutationObserver(debouncedScrollToBottom);
observer.observe(messagesContainer, { childList: true, subtree: true });