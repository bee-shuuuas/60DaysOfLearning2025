function setupDetails(action, id) {
    // Wrap the async function in an await and a runtime.sendMessage with the result
    // This should always call runtime.sendMessage, even if an error is thrown
    const wrapAsyncSendMessage = action =>
        `(async function () {
    const result = { asyncFuncID: '${id}' };
    try {
        result.content = await (${action})();
    }
    catch(x) {
        // Make an explicit copy of the Error properties
        result.error = { 
            message: x.message, 
            arguments: x.arguments, 
            type: x.type, 
            name: x.name, 
            stack: x.stack 
        };
    }
    finally {
        // Always call sendMessage, as without it this might loop forever
        chrome.runtime.sendMessage(result);
    }
})()`;

    // Apply this wrapper to the code passed
    let execArgs = {};
    if (typeof action === 'function' || typeof action === 'string')
        // Passed a function or string, wrap it directly
        execArgs.code = wrapAsyncSendMessage(action);
    else if (action.code) {
        // Passed details object https://developer.chrome.com/extensions/tabs#method-executeScript
        execArgs = action;
        execArgs.code = wrapAsyncSendMessage(action.code);
    }
    else if (action.file)
        throw new Error(`Cannot execute ${action.file}. File based execute scripts are not supported.`);
    else
        throw new Error(`Cannot execute ${JSON.stringify(action)}, it must be a function, string, or have a code property.`);

    return execArgs;
}

function promisifyRuntimeMessage(id) {
    // We don't have a reject because the finally in the script wrapper should ensure this always gets called.
    return new Promise(resolve => {
        const listener = request => {
            // Check that the message sent is intended for this listener
            if (request && request.asyncFuncID === id) {

                // Remove this listener
                chrome.runtime.onMessage.removeListener(listener);
                resolve(request);
            }

            // Return false as we don't want to keep this channel open https://developer.chrome.com/extensions/runtime#event-onMessage
            return false;
        };

        chrome.runtime.onMessage.addListener(listener);
    });
}

chrome.tabs.executeAsyncFunction = async function (tab, action) {

    // Generate a random 4-char key to avoid clashes if called multiple times
    const id = Math.floor((1 + Math.random()) * 0x10000).toString(16).substring(1);

    const details = setupDetails(action, id);
    const message = promisifyRuntimeMessage(id);

    // This will return a serialised promise, which will be broken
    await chrome.tabs.executeScript(tab, details);

    // Wait until we have the result message
    const { content, error } = await message;

    if (error)
        throw new Error(`Error thrown in execution script: ${error.message}.
Stack: ${error.stack}`)

    return content;
}

document.addEventListener('DOMContentLoaded', async function () {
    const notInUdemy = document.querySelector("#notInUdemy")
    const inUdemy = document.querySelector("#inUdemy")
    
    chrome.tabs.query({active: true, lastFocusedWindow: true}, tabs => {
        let url = tabs[0].url;
        
        const regex = /[a-zA-Z]+:\/\/[a-zA-Z]+.udemy.com\/course\/[a-zA-Z0-9\/-]*\//
        const isInUdemyCourse = regex.test(url);

        if (isInUdemyCourse) {
            notInUdemy.classList.add("hide")
            inUdemy.classList.remove("hide");

            doProcess();

        } else {
            notInUdemy.classList.remove("hide");
            inUdemy.classList.add("hide")
        }
        
    });
})

async function doProcess() {
    const resultEl = document.querySelector("#result");
    
    const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
    const tab = tabs[0];
    let res;
    try {
        const creationDate = await chrome.tabs.executeAsyncFunction(tab.id, getCreationDate);
        resultEl.innerHTML = creationDate;
    } catch (e) {
        return;
    }
}

async function getCreationDate () {
    var output = ``;
	const courseId = document.querySelector("[data-clp-course-id]").getAttribute("data-clp-course-id")
	const res = await fetch(`https://www.udemy.com/api-2.0/courses/${courseId}/?fields[course]=title,image_240x135,url,is_paid,price`).then(res => res.json()).then(res => res).catch(error => output += `<span class="error">Couldn't get course data!</span>`)
    const id = res.id;

    if(id) {
       const title = res.title
       const image = res.image_240x135
       const url = res.url
       const is_paid = res.is_paid
       const price = res.price

       output += `<img src='${image}' width="100%" alt="course image"><div class="price-wrapper">Original Price:<br>`
       is_paid ? output += `<span class="price">${price}</span>` : output += `<span class="price" style="color:green">Free</span>`
       is_paid ? output += `</div><a href="https://infognu.com/go?link=https://www.udemy.com${url}" target="_blank"><button class="button">Get discount (up to 90%)</button></a>` : ``;



    }

    return output;
}

