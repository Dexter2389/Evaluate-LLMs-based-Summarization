import requests
import re
import json
import uuid
import unicodedata
from bs4 import BeautifulSoup
from bs4.element import PageElement


def convert_math_to_latex_string(child: PageElement):
    if child.name == "math":
        return unicodedata.normalize("NFKD", child["alttext"])
    elif child.name == "span" or child.name == "a" or child.name == "cite":
        if child["class"] == ["ltx_note", "ltx_role_footnote"]:
            footnote = "".join(
                [
                    unicodedata.normalize("NFKD", sub_child)
                    if isinstance(sub_child, str)
                    else convert_math_to_latex_string(sub_child)
                    for sub_child in child.find_next(
                        "span", {"class": "ltx_note_content"}
                    ).children
                ][2:]
            )
            return f"({footnote})"
        elif child["class"] == ["ltx_text"]:
            return "".join(
                [
                    unicodedata.normalize("NFKD", sub_child)
                    if isinstance(sub_child, str)
                    else convert_math_to_latex_string(sub_child)
                    for sub_child in child.children
                ]
            )
        elif child["class"] == ["ltx_equation", "ltx_eqn_table"]:
            return ""
        return unicodedata.normalize("NFKD", child.text)
    else:
        return ""


def parse_section_paragraphs(section):
    paragraphs = []
    for paras in section:
        for para in paras.children:
            if para.name == "p":
                paragraphs.extend(
                    [
                        unicodedata.normalize("NFKD", child)
                        if isinstance(child, str)
                        else convert_math_to_latex_string(child)
                        for child in para.children
                    ]
                )
            elif para.name == "table":
                paragraphs.append(
                    convert_math_to_latex_string(
                        para.find_next("math", {"class": "ltx_Math"})
                    )
                )

    return re.sub(
        r"\b([a-zA-Z0-9]+)\s*{[^}]*}",
        r"\1",
        " ".join(paragraphs),
    )


def parse_section_title_or_abstract_text(section, heading_tag):
    return re.sub(
        r"\b([a-zA-Z0-9]+){[^}]*}",
        r"\1",
        "".join(
            [
                unicodedata.normalize("NFKD", child).strip()
                if isinstance(child, str)
                else convert_math_to_latex_string(child)
                for child in section.find(heading_tag).children
            ]
        ),
    )


def scrap_paper(url):
    parsed_paper = {}
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")

    article = soup.find("article", {"class": "ltx_document"})
    parsed_paper["title"] = parse_section_title_or_abstract_text(article, "h1")

    parsed_paper["summary"] = parse_section_title_or_abstract_text(
        article.find("div", {"class": "ltx_abstract"}), "p"
    )

    document_sections = []
    sections = article.find_all("section", {"class": "ltx_section"})
    for section in sections:
        document_sections.append(
            {
                "subtitle": parse_section_title_or_abstract_text(section, "h2"),
                "text": parse_section_paragraphs(
                    section.find_all("div", {"class": "ltx_para"})
                ),
            }
        )

        subsections = section.find_all("section", {"class": "ltx_subsection"})
        if len(subsections) > 0:
            for subsection in subsections:
                document_sections.append(
                    {
                        "subtitle": parse_section_title_or_abstract_text(
                            subsection, "h3"
                        ),
                        "text": parse_section_paragraphs(
                            subsection.find_all("div", {"class": "ltx_para"})
                        ),
                    }
                )

                subsubsections = subsection.find_all(
                    "section", {"class": "ltx_subsubsection"}
                )
                if len(subsubsections) > 0:
                    for subsubsection in subsubsections:
                        document_sections.append(
                            {
                                "subtitle": parse_section_title_or_abstract_text(
                                    subsubsection, "h4"
                                ),
                                "text": parse_section_paragraphs(
                                    subsubsection.find_all("div", {"class": "ltx_para"})
                                ),
                            }
                        )
        subsection_as_paragraph = section.find_all(
            "section", {"class": "ltx_paragraph"}
        )
        if len(subsection_as_paragraph) > 0:
            for subsection in subsection_as_paragraph:
                document_sections.append(
                    {
                        "subtitle": parse_section_title_or_abstract_text(
                            subsection, "h4"
                        ),
                        "text": parse_section_paragraphs(
                            subsection.find_all("div", {"class": "ltx_para"})
                        ),
                    }
                )
    parsed_paper["document"] = document_sections
    parsed_paper["url"] = url
    parsed_paper["id"] = str(uuid.uuid4())
    return parsed_paper


URLS = [
    "https://ar5iv.labs.arxiv.org/html/2305.04856",
    "https://ar5iv.labs.arxiv.org/html/2305.04854",
    "https://ar5iv.labs.arxiv.org/html/2305.04866",
    "https://ar5iv.labs.arxiv.org/html/2305.04868",
    "https://ar5iv.labs.arxiv.org/html/2305.04869",
    "https://ar5iv.labs.arxiv.org/html/2305.04870",
    "https://ar5iv.labs.arxiv.org/html/2305.04923",
    "https://ar5iv.labs.arxiv.org/html/2305.04924",
    "https://ar5iv.labs.arxiv.org/html/2305.04925",
    "https://ar5iv.labs.arxiv.org/html/2305.05658",
    "https://ar5iv.labs.arxiv.org/html/2305.05661",
    "https://ar5iv.labs.arxiv.org/html/2305.05664",
    "https://ar5iv.labs.arxiv.org/html/2305.05665",
    "https://ar5iv.labs.arxiv.org/html/2305.05780",
    "https://ar5iv.labs.arxiv.org/html/2305.05778",
    "https://ar5iv.labs.arxiv.org/html/2305.07824",
    "https://ar5iv.labs.arxiv.org/html/2305.09025",
    "https://ar5iv.labs.arxiv.org/html/2305.09021",
    "https://ar5iv.labs.arxiv.org/html/2305.09018",
    "https://ar5iv.labs.arxiv.org/html/2305.09013",
    "https://ar5iv.labs.arxiv.org/html/2310.11628",
    "https://ar5iv.labs.arxiv.org/html/2310.11627",
    "https://ar5iv.labs.arxiv.org/html/2310.11626",
    "https://ar5iv.labs.arxiv.org/html/2310.11134",
    "https://ar5iv.labs.arxiv.org/html/2310.11131",
    "https://ar5iv.labs.arxiv.org/html/2310.11128",
    "https://ar5iv.labs.arxiv.org/html/2310.16172",
    "https://ar5iv.labs.arxiv.org/html/2310.16110",
    "https://ar5iv.labs.arxiv.org/html/2310.13658",
    "https://ar5iv.labs.arxiv.org/html/2310.13659",
    "https://ar5iv.labs.arxiv.org/html/2310.13662",
]

# a = [scrap_paper(URL) for URL in URLS]
# with open("papers.json", "w", encoding="utf-16") as f:
#     json.dump(a, f, indent=4)
