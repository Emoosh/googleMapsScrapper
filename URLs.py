from enum import Enum

class URL(Enum):
    GOOGLE_MAPS_BASE_URL= "https://www.google.com/maps/search/{keyword}/@{latitude},{longitude},15z"

    def format_url(self, **kwargs):
        return self.value.format(**kwargs)