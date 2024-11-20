import pickle
import os

def save_object(filepath, obj):
    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path, exist_ok=True)

        with open(filepath, 'wb') as file:
            pickle.dump(obj, file)

    except Exception as e:
        return str(e)
