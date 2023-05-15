In order to communicate with OpenAI, you need an account on openai.com and an API key that they provide.

- Go to https://platform.openai.com

- Create an account if you don’t have one

- Click on the account dropdown in the top right corner

- Go to View API keys

- Click the button to “Create new secret key”
- Save your secret key in your private notes:

On Windows:

- Use the search bar in the Start menu to find “Edit the system environment variables”.

- Click “Environment variables”

- Use the upper “New…” button to add a User variable

- Create a new variable called OPENAI_API_KEY and set the value to the secret key you got from your account settings on openai.com

## Debug when openai could not be imported 
```
import sys
print(sys.executable)
C:\WorkSpace\pytest10\.venv\Scripts\python.exe -m pip install openai
```