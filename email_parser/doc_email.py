import regex
import pandas as pd

from . import nlp



class Email:

    def __init__(self,
                 raw_text):
        """ Constructor for email
        :param raw_text: raw text of email
        """
        self.raw_text = raw_text
        self.list_emails = self.f_split_email_thread()

    def f_split_email_thread(self):
        """ Function to split a thread of email into a list of individual email.

        Two main formats of header are recognized:

        1) Multi-lines header similar to
                De : sads Cadsfdsf [mailto:sdadsad@google.ca]
                Envoyé : 30 mars 2015 08:33
                À : asdsad, sadsadasd (CA - asdasd)
                Objet : Re: TR: sadasdasdsad sa dsa
        2) Le 2015-03-30 à 08:25, Luc, Archambault (CA - Drummondville) <larchambault@google.ca> a écrit :

        Returns:
            list of dict. Dict contains for each email: (body, header, start, start_header, date, lang)

        """

        pattern = r"(((\n{1}\t*|\n(-{4,}.*-{4,}\s*)|^)(([> *]*(de|from|Exp.diteur|Subject)[\s]*:).*(\n[^A-Z].*)?[\r\n\t\s,]{1,}){1,})(([> *\t]*[\p{L}\p{M}' -]*[\s]*:).*((\n[ ]{3,7}?.*|(\n<.*))*)[\r\n\t\s,]{1,3}?){2,}" \
                  r"|(\s*((((de|from|Exp.diteur|Subject)[\s]*:).{0,200}?[\r\n\t\s,]{1,}){1})(?!de)(((envoy.|.|to|date).?[\s]*:).*?){1,}(((objet|subject)[\s]*:).*?[!?.><,]){1})" \
                  r"|((?<=\n)(([ >\t]*)(le|on|el).{0,30}\d{4,}.{0,100}\n*.{0,100}(wrote|.crit|escribió)\s*:))" \
                  r"|(\b(le|on)\s*((\d{2,4}[- ]){3}|(\d{1,2}.{1,8}\d{4}))[^\n]*?(wrote|.crit)\s*:)" \
                  r"|$)"

        results = regex.finditer(pattern, self.raw_text, flags=regex.IGNORECASE)
        start_of_current_header = 0
        end_of_current_header = 0
        part_email = 1

        if results is not None:
            list_email = []

            for result in results:

                start_of_next_header = result.start()

                # if header_group is not None and full_email[0:header_group.start()].lstrip() == "":
                if start_of_current_header != end_of_current_header:
                    header = self.raw_text[start_of_current_header: end_of_current_header]
                    body = self.raw_text[end_of_current_header:start_of_next_header]

                    start = end_of_current_header
                    start_header = start_of_current_header

                # Case where no header was found (either last email of thread or regex didn't find it)
                else:
                    header = ""
                    body = self.raw_text[end_of_current_header:start_of_next_header]
                    start = end_of_current_header
                    start_header = start_of_current_header


                #  we detect language for each email of the thread and default to detected thread language otherwise
                # We detect only on first 150 characters
                lang = nlp.f_detect_language(body[:150])

                if body.strip() != "" or header != "":
                    list_email.append({"body": body,
                                       "header": header,
                                       "start": start,
                                       "start_header": start_header,
                                       "lang": lang,
                                       "part": part_email
                                       })
                    part_email += 1
                # previous_from_tag = current_from_tag
                start_of_current_header = result.start()
                end_of_current_header = result.end()

            return list_email
        # Case were mail is not a thread
        else:
            return [{"body": self.raw_text,
                     "header": "",
                     "start": 0}]

    @staticmethod
    def f_find_person_in_header(header, df_result=pd.DataFrame()):
        results = []
        dict_header = Email.f_split_email_headers(header)
        for key in ["to", "cc", "from"]:
            if key in dict_header.keys():
                line_header = dict_header[key][0]
                start_posit = dict_header[key][1]
                pattern_person = r"(?<=\s|'|^)[\p{L}\p{M}\s,-]{2,}(?=[\s;']|$)"
                list_results = regex.finditer(pattern_person, line_header, flags=regex.IGNORECASE)
                for match in list_results:
                    value = match.group()
                    if value.strip() != "":
                        start = match.start()
                        end = match.end()
                        results.append(["PER",
                                          value,
                                          start_posit + start,
                                          start_posit + end,
                                          1
                                          ])
        df_result = nlp.f_concat_results(df_result, results)
        return df_result

    @staticmethod
    def f_split_email_headers(header):
        """ SPlit headers in from/to/date,...in a dictionnary

        Args:
            header:

        Returns:

        """
        matching_header_keywords = {"à": "to",
                                    "Destinataire": "to",
                                    "de": "from",
                                    "envoyé": "date",
                                    "sent": "date",
                                    "objet": "subject"}
        dict_results = {}
        pattern = r"((?<=\s|^)(à|À|a\p{M}|Cc|To|De|From|Envoy.|Date|Sent|Objet|Subject|Destinataire)\s?:)[ ]*((.*?)[ ]*((\n[ ]{3,7}?.*)*))(?=[\p{L}\p{M}]*\s{1,}:| > |\n|$)"
        list_results = regex.finditer(pattern, header, flags=regex.IGNORECASE)
        for match in list_results:
            key_word = match.group(2).strip().lower()
            key_word_matched = matching_header_keywords.get(key_word)
            dict_results[key_word_matched if not key_word_matched is None else key_word] = [match.group(3),
                                                                                            match.span(3)[0],
                                                                                            match.span(3)[1]]
        return dict_results
