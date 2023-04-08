class ConfigObject(object):
    def __init__(self, params_dict=None):
        self.params_dict = params_dict if params_dict is not None else dict()

    def __setattr__(self, key, value):
        if key == "params_dict":
            object.__setattr__(self, key, value)
        else:
            self.params_dict[key] = value

    def __getattr__(self, key):
        return self.params_dict[key]
