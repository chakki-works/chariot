import os
import sys
import re
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from src.utils.storage import Storage, CsvFile


def format(version):
    """
    Format cosme domain file
    """

    # Set file names
    source_file = CsvFile.create("interim", "cosme", "merged", version)
    target_file = source_file.convert(attribute_to="formatted")

    if not source_file.exists():
        raise Exception("Merged file has not created yet.")

    # Format source file to target file
    ex_checker = ExcludeChecker()
    formatter = Formatter()
    with open(target_file.path, mode="w", encoding="utf-8") as f:
        count = 0
        ignored = 0
        for element in source_file.fetch(delimiter="\t"):
            if count == 0:
                line = "\t".join(source_file.header) + "\n"
                f.write(line)
                count += 1
                continue

            element = formatter.apply(element)
            ignore_reason = ex_checker.should_ignore(element)
            if ignore_reason:
                ignored += 1
                """
                desc = "\t".join([ignore_reason, element["catchphrase"],
                                  element["description"]])
                print(desc)
                """
            else:
                if element["description"].startswith("【注】 "):
                    print(element)
                    break
                f.write(source_file.to_line(element))
                count += 1

    print("{} record formatted ({} is ignored).".format(count, ignored))


class Formatter():

    def __init__(self):
        pass

    def format(self, input_text):
        raise Exception("Subclass have to implements format method.")


class CosmeFormatter(Formatter):

    def __init__(self):
        super().__init__()
        self.sentence_pattern = re.compile(".+?。")
        self.notation_pattern = re.compile("【.+?】")
        self.strong_pattern = re.compile("＜.+?＞")
        self.reference_pattern = re.compile("[*|＊|※][0-9０-９]")
        self.number_pattern = re.compile("[0-9０-９][\.|．]")

    def format(self, input_text):
        return self.extract_valid_sentence(input_text)

    def apply(self, element):
        applied = self._apply(element, ["description"],
                              self.extract_valid_sentence)
        applied = self._apply(applied, ["catchphrase"],
                              self.base_strip)
        applied = self._apply(applied, ["catchphrase"],
                              self.strip)

        return applied

    def _apply(self, elements, key, func):
        applied = {}
        for e in elements:
            if e in key:
                applied[e] = func(elements[e])
            else:
                applied[e] = elements[e]
        return applied

    def extract_valid_sentence(self, text):
        sentences = []
        for matched in re.findall(self.sentence_pattern, text):
            s = self.base_strip(matched)
            if self.is_valid(s):
                sentences.append(self.strip(s))
            else:
                # ignore after non-valid sentence
                break
        return "".join(sentences)

    def is_valid(self, sentence):
        caution_char = ["【注】", "※"]
        context = ["実際の商品と写真は", "ご了承下さい",
                   "数量限定", "取扱店希少"]
        for c in caution_char:
            if sentence.startswith(c):
                return False
        for c in context:
            if c in sentence:
                return False

        return True

    def base_strip(self, sentence):
        # Exclude disc
        stripped = sentence.replace("■", "").replace("●", "").replace("・", "")
        stripped = re.sub(self.number_pattern, "", stripped)

        # Exclude reference
        stripped = re.sub(self.reference_pattern, "", stripped)

        # Format zenkaku char
        stripped = stripped.replace("”", "\"").replace("！", "!")
        stripped = stripped.replace("％", "%").replace("＆", "&")
        stripped = stripped.replace("『", "「").replace("』", "」")
        return stripped.strip()

    def strip(self, sentence):
        stripped = re.sub(self.notation_pattern, "", sentence)
        stripped = re.sub(self.strong_pattern, "", stripped)
        stripped = stripped.replace("*", "").replace("＊", "")
        return stripped.strip()


class ExcludeChecker():

    def __init__(self):
        self.stocked = []

    def should_ignore(self, element):
        reason = ""
        if not element["catchphrase"] or not element["description"]:
            reason = "catch phrase or description does not exist."

        if len(element["description"]) < 10:
            reason = "too short description"

        if element["catchphrase"] in self.stocked:
            reason = "duplicate catch phrase"
        else:
            self.stocked.append(element["catchphrase"])

        return reason


if __name__ == "__main__":
    format("20180524")
