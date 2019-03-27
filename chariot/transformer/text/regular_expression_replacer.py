from chariot.transformer.text.base import TextFilter
import re


class RegularExpressionReplacer(TextFilter):

    def __init__(self, pattern, replacement, copy=True):
        super().__init__(copy)
        self.pattern = pattern
        self.replacement = replacement

    def apply(self, text):
        return re.sub(self.pattern, self.replacement, text)
