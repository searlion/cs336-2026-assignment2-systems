# word_count.py: words of prose in the article, without code blocks, {% details %} boxes,
# KaTeX blocks, images, anchors and URLs (link text is counted). Tables are counted.
import re
import sys

s = open(sys.argv[1]).read()
s = re.sub(r"```.*?```", " ", s, flags=re.S)
s = re.sub(r"\{% details.*?\{% enddetails %\}", " ", s, flags=re.S)
s = re.sub(r"\{% katex %\}.*?\{% endkatex %\}", " ", s, flags=re.S)
s = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", s)
s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)
s = re.sub(r"</?a[^>]*>", " ", s)
s = re.sub(r"^\|[-| ]+\|$", " ", s, flags=re.M)
words = re.findall(r"[A-Za-z0-9][\w.'’×/+-]*", s)
print(len(words), "words;", f"{len(words) / 230:.1f} min at 230 wpm")
