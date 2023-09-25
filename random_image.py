import requests, clipboard, pyautogui, json, random

image_types = {
    "Cat":["https://aws.random.cat/meow","file"],
    "Dog":["https://random.dog/woof.json","url"],
    "Fox":["https://randomfox.ca/floof/","image"]
    }

image_type = random.choice(list(image_types.keys()))

clipboard.copy(json.loads(requests.get(image_types[image_type][0]).content.decode("utf8"))[image_types[image_type][1]])
pyautogui.typewrite(f"**Random {image_type} Photo**")
pyautogui.hotkey("shift","enter")
pyautogui.hotkey("ctrl","v")
pyautogui.press("enter") 
