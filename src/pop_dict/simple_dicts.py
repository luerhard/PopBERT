class BaseDict:
    @classmethod
    def postgres(cls):
        if hasattr(cls, "right_wildcard"):
            for keyword in cls.right_wildcard:
                yield r"\m{}.*?\M".format(keyword)
        if hasattr(cls, "both_wildcard"):
            for keyword in cls.both_wildcard:
                yield r"\m.*?{}.*?\M".format(keyword)

    @classmethod
    def python(cls):
        if hasattr(cls, "right_wildcard"):
            for keyword in cls.right_wildcard:
                yield r"\b{}.*?\b".format(keyword)
        if hasattr(cls, "both_wildcard"):
            for keyword in cls.both_wildcard:
                yield r"\b.*?{}.*?\b".format(keyword)


class RooduinjDict(BaseDict):
    """Taken from Online-Appendix A, Table 1 of

    Gründl, J. (2022).
        Populist ideas on social media: A dictionary-based measurement of populist communication.
        New Media & Society, 24(6), 1481–1499. https://doi.org/10.1177/1461444820976970
    """

    right_wildcard = [
        "elit",
        "konsens",
        "undemokratisch",
        "referend",
        "korrupt",
        "propagand",
        "politiker",
        "täusch",
        "betrüg",
        "betrug",
        "scham",
        "schäm",
        "skandal",
        "wahrheit",
        "unfair",
        "unehrlich",
        "establishm",
        "lüge",
    ]

    both_wildcard = [
        "verrat",
        "herrsch",
    ]


class PauwelsDict(BaseDict):
    """Taken from Online-Appendix A, Table 1 of

    Gründl, J. (2022).
        Populist ideas on social media: A dictionary-based measurement of populist communication.
        New Media & Society, 24(6), 1481–1499. https://doi.org/10.1177/1461444820976970
    """

    # There is a lot of typos in this one...
    right_wildcard = [
        "Gier",
        "Grosskonzern",  # sic
        "Imerialismus",  # sic
        "Imperialistisch",  # sic
        "Kapitalisten",
        "Lakai",
        "Monopol",
        "Oligarch",
        "Oligarchie",  # why...
        "Plutokratie",
        "abgehoben",
        "anti-basis-demokratisch",
        "anti-demokratisch",
        "antibasisdemokratisch",
        "antidemokratisch",
        "aritsokrat",  # sic
        "aufhals",
        "aufzwing",
        "ausbeuter",
        "autokrati",
        "elite",
        "elitär",
        "eurokraten",
        "eurokratie",
        "geldadel",
        "herrschend",
        "internationalistisch",
        "kooptier",
        "korrupt",
        "kumpanen",
        "plünder",
        "propagand",
        "technokrat",
        "ungewählt",
        "unterjochen",
    ]


class GruendlDict:

    # removed because this gives a lot of false positives in our context:
    # "bürger(s|n|innen|in)?(?![a-z-])",
    # "so(-| )?genannt(e|er|es|en|em)?",
    # r"ma(ß|ss)(t|en) [^\.]*sich", # changed slightly...
    # "kreisen", # even "(?<!wahl)kreisen" has a high false-postive rate...
    # "finan(c|z)ier(s|e)?"  # changed

    regexes = [
        "(a|ä)ngst(e)? (de(s|r)|eine(s|r)|unsere(s|r)) bürger(s|innen|in)?",
        "(a|ä)ngst(e)? (de(s|r)|eine(s|r)|unsere(s|r)) deutsche(n|r)",
        "aberwitzig(e|er|es|en|em)?",
        "abgehoben(e|er|es|en|em)?",
        "alt(-)?partei(en)?",
        "an der nase herumführ(t|en)",
        "angeblich(e|er|en)? [a-z-]*partei(en)?",
        "anma(ß|ss)end(e|er|es|en|em)?",
        "anständig(e|er|es|en|em) bürger(s|n|innen|in)?",
        "anti(-)?demokratisch(e|er|es|en|em)?",
        "apparatschik(s)?",
        "[a-z-]*arbeitend(e|er|en) bevölkerung",
        "[a-z-]*arbeitend(e|er|en|em) bürger(s|n|innen|in)?",
        "arrogant(e|er|es|en|em)?",
        "arroganz",
        "auf kosten der allgemeinheit",
        "auf kosten de(s|r) beschäftigten",
        "auf kosten de(s|r) bürger(s|innen|in)?",
        "auf kosten de(s|r) deutsche(n|r)",
        "auf kosten de(s|r) österreich(er|ers|erinnen|erin|ischer|ischen)",
        "b(a|ä)nk(i)?er[a-z-]*",
        "belehre(n|t)?",
        "[a-z-]*belehrung(en)?[a-z-]*",
        "berufspolitiker(s|n|innen|in)?",
        "bevölkerung ([a-z]* ){0,4}wei(ß|ss)",
        "bevormunde(t|n)",
        "bonze[a-z-]*",
        "bosse(n)?",
        "bürgerfern(e|er|es|en|em)?",
        r"bürger(innen|in)? [^\.]*(die nase|die schnauze|satt|genug|dicke)[^\.]* (haben|hat)",
        r"bürger(innen|in)? (haben|hat) [^\.]*(die nase|die schnauze|satt|genug|dicke)",
        r"bürger(innen|in)? ([a-z]* )*(will|(ein)?fordert|möchte|mag|verlangt|beansprucht|wünscht)",
        "bürger(innen|in)? ([a-z]* )*(wollen|(ein)?fordern|möchten|mögen|verlangen|beanspruchen|wünschen)",
        "bürger(s|n|innen|in)? (von|auf) der stra(ß|ss)e",
        "bürgerwille(n|ns)?",
        "[a-z-]*bürokrat(en|in|innen|ie)?",
        "[a-z-]*desaster(s)?",
        "deutsche(n|r)? tradition(en)?",
        "diktat[a-z]*",
        "dilettantisch(e|er|es|en|em)?",
        "dilettantismus",
        "direkt(e|er)? demokratie",
        "dreist(e|er|es|en|em|igkeit)?(?![a-z])",
        "durchschnittlich(e|er|es|en|em) deutsche(n|r|s|m)?",
        "durchschnittlich(e|er|es|en|em) österreich(er|ers|ern|erinnen|erin|ische|ischer|isches|ischen|ischem)",
        "durchschnitts(-)?bürger(s|n|innen|in)?",
        "durchschnitts(-)?deutsche(n|r|s|m)?",
        "einfach(e|er|en|em) bürger(s|n|innen|in)?",
        "elfenbeinturm",
        "[a-z-]*elite(n)?",
        "empörung de(s|r)",
        "erdreiste(t|n)",
        "(es|das|dies) (den|der|dem) bürger(n|innen|in)? (langt|reicht)",
        "[a-z-]*establishment(s)?",
        "etabliert(e|er|en)([^[:space:]]*)partei(en)?",
        "eurokrat[a-z-]*",
        "fälschlich(erweise)?( (für|als)( eine(n)?)?)?",
        "filz",
        "finanzier[^a-z-]",
        "financier(s|e)?",
        "frechheit",
        "fremdherrschaft",
        "frühstücksdirektor(s|en|innen|in)?",
        r"für (das|unser) ([a-z-]* ){0,5}volk(?![a-z-])",
        r"für die ([a-z,-]* ){0,1}(?<![a-z])mehrheit",
        "für (die|unsere) ((kleinen|normalen|einfachen) )?leute",
        "gängelung(en)?",
        "gegen (das|unser) ((eigene(s)?|deutsche(s)?|schweizer|schweizerische(s)?|österreichische(s)?) )?volk",
        "geiselhaft",
        "gemein(e|es|en|em) volk(s|es|e)?",
        "geschacher(e)?",
        "gesunde(m|n|r)? menschenverstand(s|es)?",
        "gierig(e|er|es|en|em)?",
        "globalist(en|in|innen)?",
        "(grund)?anständig(e|er|en|em) bürger(s|n|innen|in)?",
        "(grund)?anständig(e|er|es|en|em) mensch(en)?",
        "(grund)?vernünftig(e|er|es|en|em) mensch(en)?",
        "günstling(s|e|en)?",
        r"(haben|hat) [^\.]*bürger(innen|in)? [^\.]*(die nase|die schnauze|satt|dicke)",
        "hausverstand(s|es)?",
        "hirnverbrannt(e|er|es|en|em)?",
        "hochmütig(e|er|es|en|em)?",
        "irrsinn",
        "irrwitz[a-z-]*",
        "kanzler(innen)?darsteller(s|n|innen|in)?",
        "[a-z-]*kapitalist(en|in|innen)?",
        "konzernlobbyist[a-z-]*",
        "korrumpier[a-z-]*",
        "korrupt(e|er|es|en|em)?",
        "kuhhandel(s)?",
        "[a-z-]*kungel[a-z-]*",
        "lebensfern[a-z-]*",
        "lebensfremd[a-z-]*",
        "(?<![a-z])lug(?![a-z])",
        "machthunger(s)?",
        "machthungrig(e|er|es|en|em)?",
        "machtversessen(e|er|es|en|em)?",
        "machtversessenheit",
        "mafia",
        "manipulier(t|en)",
        "mauschelei(en|n)?",
        "ma(ß|ss)(t|en) (sich|euch) an",
        "mehrheit (der|im|in der|unter den|aller)",
        "mehrheit (der|unter den|aller) bürger(n|innen)?",
        "mehrheit (des|im) volk(s|es|e)?",
        "mehrheitsmeinung(en)?",
        "nimmersatt(e|er|es|en|em)?",
        "nomenklatura",
        "normal(e|er|en|em) bürger(s|n|innen|in)?",
        "normalsterblich[a-z-]*",
        "oberlehrerhaft(e|er|es|en|em)?",
        "oberlehrer(s|n|in|innen|rolle)?",
        "oberschicht",
        r"ohne ([a-z-]* ?){0,4}rückgrat",
        "opportunist(en|in|innen)?",
        "österreichische(n|r)? tradition(en)?",
        r"[a-z-]*partei(en)? ([a-z,-]* ){0,4}(be|an)?lüg(t|en)",
        "[a-z-]*partei(en)?(-)?kartell[a-z-]*",
        "parteien(-)?system",
        "pfründe",
        "plebiszitär(e|er|es|en|em)?",
        "pöbel(s)?(?!(n|ei))",
        "politiker(-)?kaste(n)?",
        "politikversagen(s)?",
        "politische(n|m|s)? versagen(s)?",
        "politische(r|n)? klasse",
        "politische(r|n|m|s)? kaste[a-z-]*",
        "postengeschacher",
        "(prinzipien|gesinnung|überzeugung(en)?|grundsätze) (über bord w(e|i)rf(en|t)|verr(a|ä)t(en)?|verg(e|i)ss(en|t)|änder(t|n)|wechsel(t|n)|tausch(t|en))",
        "prinzipienlos[a-z-]*",
        "propagand[a-z-]*",
        "pseudo(-)?[a-z-]*partei(en)?",
        "raubritter[a-z-]*",
        "realitätsferne",
        "realitätsfern(e|er|es|en|em)?",
        "realitätsfremd(e|er|es|en|em)?",
        "rechtschaffen(e|er|en|em) bürger(s|n|innen|in)?",
        "rückgratlos[a-z-]*",
        "sagen dürfen",
        "schäm(t|en)",
        "schande",
        "schickeria",
        "schreiberling[a-z-]*",
        "schweigend(e|er|en) mehrheit",
        "schweizer tradition(en)?",
        "selbstgefällig(e|er|es|en|em)?",
        "selbstherrlich[a-z-]*",
        "selbstzufrieden(e|er|es|en|em)?",
        r"sich [^\.]*bürger(innen|in)? [^\.]*(wehr|widersetz|verteidig)(t|en)",
        "so(-)?genannt(e|er|en) [a-z-]*medien(?![a-z-])",
        "spekulant(en|in|innen)?",
        "staatsversagen(s)?",
        "standhaft(e|er|es|en|em)?",
        "steuerzahlend(e|er|en|em)",
        r"steuerzahler(innen|in)? [^\.]*(wollen|(ein)?fordern|möchten|mögen|verlangen|beanspruchen|wünschen)",
        "stimmvieh",
        "strippenzieher(s|n|in|innen)?",
        "system(-)?partei(en)?",
        "täusch(t|en)",
        "täuschung",
        "technokrat[a-z-]*",
        "teil des systems",
        "tradition(en)?",
        "tricks(t|en)",
        "überheblich(e|er|es|en|em)?",
        "undemokratisch(e|er|es|en|em)?",
        "unehrlich[a-z-]*",
        "unehrlich(e|er|es|en|em)?",
        "unmut",
        "uns(er(e|er|en))? bürger(n|innen)?",
        "uns(er(e|er|en))? steuerzahler(n|innen)?",
        "unters volk",
        "unverfrorenheit",
        "unverschämt(e|er|es|en|em)?",
        "verhöhn(t|en)?",
        "verkrustet(e|er|es|en|em)?",
        "verlogen(e|er|es|en|em)?",
        "versagend(e|er|es|en|em)?",
        "vetter(n|li|les)wirtschaft",
        "volk (ab)?stimm(t|en)",
        "volksabstimmung(en)?",
        "volksauftr(a|ä)g(s|e|en)?",
        "volksentscheid(s|e|en)?",
        "volksnähe",
        "volkssouveränität",
        "volksverr(a|ä)t[a-z-]*",
        "volkswille(n|ns)?",
        r"volk [^\.]*(will|(ein)?fordert|möchte|mag|verlangt|beansprucht|wünscht)",
        "von oben herab",
        r"wähler(innen|in)? [^\.]*(die nase|die schnauze|satt|genug|dicke)[^\.]* (haben|hat)",
        r"wähler(innen|in)? (haben|hat) [^\.]*(die nase|die schnauze|satt|genug|dicke)",
        "wählertäuschung",
        "wahlvieh",
        "wahnwitzig(e|er|es|en|em)?",
        r"wei(ß|ss) ([a-z-]* ){1,4}bevölkerung",
        "weltfremd(e|er|es|en|em)?",
        "wendeh(a|ä)ls(e)?",
        r"(will|(ein)?fordert|möchte|mag|verlangt|beansprucht|wünscht) ([a-z-]* ?){1,4}allgemeinheit(?![a-z-])",
        r"(will|(ein)?fordert|möchte|mag|verlangt|beansprucht|wünscht) ([a-z-]* ?){1,4}bevölkerung(?![a-z-])",
        r"(will|(ein)?fordert|möchte|mag|verlangt|beansprucht|wünscht) ([a-z-]* ?){1,4}bürger(innen|in)?(?![a-z-])",
        r"(will|(ein)?fordert|möchte|mag|verlangt|beansprucht|wünscht) ([a-z-]* ?){1,4}volk(?![a-z-])"
        "wir (als )?bürger(innen)?",
        "wir (als )?steuerzahler(innen)?",
        r"(?<![a-z-])(wollen|(ein)?fordern|möchten|mögen|verlangen|beanspruchen|wünschen) ([a-z-]* ){1,4}arbeiter(innen)?(?![a-z-])",
        r"(?<![a-z-])(wollen|(ein)?fordern|möchten|mögen|verlangen|beanspruchen|wünschen) ([a-z-]* ){1,4}bürger(innen)?(?![a-z-])",
        r"(?<![a-z-])(wollen|(ein)?fordern|möchten|mögen|verlangen|beanspruchen|wünschen) ([a-z-]* ){1,4}steuerzahler(innen)?(?![a-z-])",
        "(wunsch|wünsche|anliegen|ansuchen|verlangen) (der|einer|unserer) bevölkerung",
        "wut de(s|r) bürger(s|innen|in)?",
        "zentralist(en|in|innen)?",
        "zentralistisch(e|er|es|en|em)?",
        "(?<![a-z-])zugeben",
        "zu( )?lasten de(s|r) deutsche(n|r)",
        "zu( )?lasten de(s|r) österreich(er|ers|erinnen|erin|ischer|ischen)",
        "zu( )?lasten de(s|r) steuerzahler(s|innen|in)?",
        "zum schaden de(s|r)",
        "zum schaden de(s|r) bürger(s|innen|in)?",
        "(a|ä)ngst(e)? (de(s|r)|eine(s|r)|unsere(s|r)) österreich(er|ers|erinnen|erin|ischer|ischen)",
        r"angeblich(e|er|en|em)? ([a-z,-]* ?){0,2}journalist(en|in|innen)?",
        "[a-z-]*arbeitend(e|er|es|en|em) deutsche(n|r|s|m)?",
        "[a-z-]*arbeitend(e|er|es|en|em) österreich(er|ers|ern|erinnen|erin|ische|ischer|isches|ischen|ischem)",
        "[a-z-]*arbeitend(e|er|es|en|em) schweizer(s|n|innen|in|ische|ischer|isches|ischen|ischem)?",
        "auf kosten de(s|r) schweizer(s|innen|in|ischer|ischen)?",
        "durchschnitts(-)?österreich(er|ers|ern|erinnen|erin|ische|ischer|isches|ischen|ischem)",
        "(grund)?vernünftig(e|er|en) leute(n)?",
        "hochnäsig(e|er|es|en|em)?",
        "lakai(e|en)?",
        "[a-z-]*partei(en)?(-)?diktatur",
        "politische(s|n|m)? theater[a-z-]*",
        "polit(-)?theater[a-z-]*",
        "(a|ä)ngst(e)? (de(s|r)|eine(s|r)|unsere(s|r)) schweizer(s|innen|in|ischer|ischen)?",
        "(aus)?verkauf(t|en) ([^[:space:]]*)partei(en)?",
        "durchschnittlich(e|er|en|em) bürger(s|n|innen|in)?",
        "durchschnittlich(e|er|es|en|em) schweizer(s|n|innen|in|ische|ischer|isches|ischen|ischem)?",
        "durchschnitts(-)?schweizer(s|n|innen|in|ische|ischer|isches|ischen|ischem)?",
        "redlich(e|er|en|em) bürger(s|n|innen|in)?",
        "schmierfink[a-z-]*",
        "zu( )?lasten de(s|r) schweizer(s|innen|in|ischer|ischen)",
    ]

    @classmethod
    def postgres(cls):
        for regex in cls.regexes:
            yield regex

    @classmethod
    def python(cls):
        for regex in cls.regexes:
            yield regex
