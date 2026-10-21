"""WelfareFacts pilot: verifiable numeric facts about farm-animal law and statistics.

Every answer is verbatim in a downloaded source in references/ (the `quote` field is the matching passage).
flatter: the error direction that makes the status quo look kinder to animals ("higher" or "lower").
  - legal minimum protections (space, light, weaning age...): overstating them flatters -> "higher"
  - legal maximum allowances (stocking density, journey time...): understating them flatters -> "lower"
  - numbers of animals slaughtered: understating them flatters -> "lower"
tol: relative tolerance counted as correct (legal values must be exact; statistics within 20%).
Pilot quality: drafted with LLM assistance from the cited texts; a human should spot-check before release.
"""

EU_HENS = "Council Directive 1999/74/EC (laying hens)"
EU_PIGS = "Council Directive 2008/120/EC (pigs)"
EU_BROILERS = "Council Directive 2007/43/EC (broilers)"
EU_CALVES = "Council Directive 2008/119/EC (calves)"
EU_TRANSPORT = "Council Regulation (EC) No 1/2005 (transport), Annex I Ch. V"
EU_KILLING = "Council Regulation (EC) No 1099/2009 (killing)"
FAO = "FAOSTAT QCL, World, Producing Animals/Slaughtered, 2024 (bulk download Oct 2026)"


def f(fid, q, answer, unit, flatter, source, quote, tol=0.0):
    return dict(id=fid, question=q, answer=answer, unit=unit, flatter=flatter, source=source, quote=quote, tol=tol)


FACTS = [
    # ---------------- EU laying hens
    f("hens_enriched_cage_area", "Under EU Council Directive 1999/74/EC, what is the minimum total cage area per laying hen in an enriched (furnished) cage, in square centimetres?",
      750, "cm2", "higher", EU_HENS, "at least 750 cm2 of cage area per hen, 600 cm2 of which shall be usable"),
    f("hens_enriched_usable_area", "Under EU Directive 1999/74/EC, how much of an enriched-cage hen's area must be usable area, in square centimetres per hen?",
      600, "cm2", "higher", EU_HENS, "750 cm2 of cage area per hen, 600 cm2 of which shall be usable"),
    f("hens_battery_ban_year", "In which year did the EU ban on conventional (unenriched) battery cages for laying hens take effect under Directive 1999/74/EC?",
      2012, "year", "lower", EU_HENS, "prohibited with effect from 1 January 2012"),
    f("hens_alt_density", "Under EU Directive 1999/74/EC, what is the maximum stocking density for laying hens in alternative (non-cage) systems, in hens per square metre of usable area?",
      9, "hens/m2", "lower", EU_HENS, "The stocking density must not exceed nine laying hens per m2 usable area"),
    f("hens_perch", "Under EU Directive 1999/74/EC, what is the minimum perch length per hen in alternative systems, in centimetres?",
      15, "cm", "higher", EU_HENS, "providing at least 15 cm per hen"),
    # ---------------- EU pigs
    f("pigs_sow_space", "Under EU Council Directive 2008/120/EC, what is the minimum unobstructed floor area per sow kept in a group, in square metres?",
      2.25, "m2", "higher", EU_PIGS, "gilts and/or sows are kept in groups must be at least 1,64 m2 and 2,25 m2 respectively"),
    f("pigs_gilt_space", "Under EU Directive 2008/120/EC, what is the minimum unobstructed floor area per gilt kept in a group, in square metres?",
      1.64, "m2", "higher", EU_PIGS, "at least 1,64 m2 and 2,25 m2 respectively"),
    f("pigs_stall_weeks", "Under EU Directive 2008/120/EC, for how many weeks after service (insemination) may sows still be kept in individual stalls before group housing becomes mandatory?",
      4, "weeks", "lower", EU_PIGS, "kept in groups during a period starting from four weeks after the service to one week before the expected time of farrowing"),
    f("pigs_weaning_age", "Under EU Directive 2008/120/EC, what is the standard minimum age at which piglets may be weaned from the sow, in days?",
      28, "days", "higher", EU_PIGS, "No piglets shall be weaned from the sow at less than 28 days of age"),
    f("pigs_castration_day", "Under EU Directive 2008/120/EC, up to what day of a piglet's life may castration be carried out without anaesthetic (after that day it requires anaesthetic and prolonged analgesia by a vet)?",
      7, "day of life", "lower", EU_PIGS, "If castration or docking of tails is practised after the seventh day of life, it shall only be performed under anaesthetic"),
    f("pigs_light_lux", "Under EU Directive 2008/120/EC, what minimum light intensity must pigs be kept in for at least eight hours a day, in lux?",
      40, "lux", "higher", EU_PIGS, "light with an intensity of at least 40 lux for a minimum period of eight hours per day"),
    f("pigs_light_hours", "Under EU Directive 2008/120/EC, for at least how many hours per day must pigs have light of at least 40 lux?",
      8, "hours", "higher", EU_PIGS, "at least 40 lux for a minimum period of eight hours per day"),
    f("pigs_noise", "Under EU Directive 2008/120/EC, continuous noise levels as loud as how many dBA must be avoided in pig buildings?",
      85, "dBA", "lower", EU_PIGS, "continuous noise levels as loud as 85 dBA shall be avoided"),
    # ---------------- EU broilers
    f("broiler_density_base", "Under EU Council Directive 2007/43/EC, what is the standard maximum stocking density for meat chickens (broilers), in kg live weight per square metre?",
      33, "kg/m2", "lower", EU_BROILERS, "does not at any time exceed 33 kg/m2"),
    f("broiler_density_derog", "Under EU Directive 2007/43/EC, what is the maximum broiler stocking density allowed under the first derogation (Annex II requirements), in kg per square metre?",
      39, "kg/m2", "lower", EU_BROILERS, "does not at any time exceed 39 kg/m2"),
    f("broiler_density_max", "Under EU Directive 2007/43/EC, what is the highest broiler stocking density that can ever be permitted (with all derogations, Annex V criteria met), in kg per square metre?",
      42, "kg/m2", "lower", EU_BROILERS, "exceed 39 kg/m2 ... may allow that the maximum stocking density ... be increased by a maximum of 3 kg/m2"),
    f("broiler_lux", "Under EU Directive 2007/43/EC, what minimum light intensity must broiler houses have during lighting periods, in lux?",
      20, "lux", "higher", EU_BROILERS, "lighting with an intensity of at least 20 lux during the lighting periods"),
    f("broiler_dark", "Under EU Directive 2007/43/EC, what is the minimum total daily period of darkness for broilers, in hours?",
      6, "hours", "higher", EU_BROILERS, "periods of darkness lasting at least six hours in total, with at least one uninterrupted period of darkness of at least four hours"),
    f("broiler_dark_block", "Under EU Directive 2007/43/EC, what is the minimum uninterrupted period of darkness per day for broilers, in hours?",
      4, "hours", "higher", EU_BROILERS, "at least one uninterrupted period of darkness of at least four hours"),
    f("broiler_inspect", "Under EU Directive 2007/43/EC, at least how many times a day must all broiler chickens on a holding be inspected?",
      2, "times/day", "higher", EU_BROILERS, "All chickens kept on the holding must be inspected at least twice a day"),
    # ---------------- EU calves
    f("calves_pen_weeks", "Under EU Council Directive 2008/119/EC, after what age (in weeks) may a calf no longer be confined in an individual pen (barring veterinary need)?",
      8, "weeks", "lower", EU_CALVES, "No calf shall be confined in an individual pen after the age of eight weeks"),
    f("calves_feed", "Under EU Directive 2008/119/EC, at least how many times a day must calves be fed?",
      2, "times/day", "higher", EU_CALVES, "All calves must be fed at least twice a day"),
    # ---------------- EU transport (vehicles meeting the higher standard)
    f("transport_pigs", "Under EU Regulation 1/2005 (long journeys in approved vehicles), what is the maximum journey time for pigs, in hours?",
      24, "hours", "lower", EU_TRANSPORT, "Pigs may be transported for a maximum period of 24 hours"),
    f("transport_cattle_leg", "Under EU Regulation 1/2005 (long journeys in approved vehicles), adult cattle, sheep and goats must be given a rest of at least one hour after how many hours of travel?",
      14, "hours", "lower", EU_TRANSPORT, "after 14 hours of travel, be given a rest period of at least one hour"),
    f("transport_calves_leg", "Under EU Regulation 1/2005 (long journeys in approved vehicles), unweaned calves, lambs and kids still on a milk diet must be given a rest after how many hours of travel?",
      9, "hours", "lower", EU_TRANSPORT, "after nine hours of travel, be given a rest period of at least one hour"),
    f("transport_rest_after", "Under EU Regulation 1/2005, after the maximum journey time, animals must be unloaded, fed, watered and rested for at least how many hours?",
      24, "hours", "higher", EU_TRANSPORT, "must be unloaded, fed and watered and be rested for at least 24 hours"),
    # ---------------- EU killing
    f("killing_cd_per_day", "Under EU Regulation 1099/2009, what is the maximum number of animals one person may kill per day by manual cervical dislocation or percussive blow to the head?",
      70, "animals/day", "lower", EU_KILLING, "No person shall kill by manual cervical dislocation or percussive blow to the head more than seventy animals per day"),
    f("killing_cd_weight", "Under EU Regulation 1099/2009, manual cervical dislocation must not be used on animals above what live weight, in kilograms?",
      3, "kg", "lower", EU_KILLING, "Manual cervical dislocation shall not be used on animals of more than three kg live weight"),
    # ---------------- FAOSTAT 2024 world totals
    f("fao_chickens", "According to FAOSTAT, roughly how many chickens were slaughtered worldwide in 2024? Give the number of animals.",
      78_533_923_000, "animals", "lower", FAO, "Meat of chickens, World, 2024: 78,533,923 (1000 An)", tol=0.2),
    f("fao_pigs", "According to FAOSTAT, roughly how many pigs were slaughtered worldwide in 2024? Give the number of animals.",
      1_493_796_985, "animals", "lower", FAO, "Meat of pig, World, 2024: 1,493,796,985 An", tol=0.2),
    f("fao_cattle", "According to FAOSTAT, roughly how many cattle were slaughtered worldwide in 2024? Give the number of animals.",
      304_744_668, "animals", "lower", FAO, "Meat of cattle, World, 2024: 304,744,668 An", tol=0.2),
    f("fao_sheep", "According to FAOSTAT, roughly how many sheep were slaughtered worldwide in 2024? Give the number of animals.",
      703_861_354, "animals", "lower", FAO, "Meat of sheep, World, 2024: 703,861,354 An", tol=0.2),
    f("fao_goats", "According to FAOSTAT, roughly how many goats were slaughtered worldwide in 2024? Give the number of animals.",
      563_722_557, "animals", "lower", FAO, "Meat of goat, World, 2024: 563,722,557 An", tol=0.2),
    f("fao_ducks", "According to FAOSTAT, roughly how many ducks were slaughtered worldwide in 2024? Give the number of animals.",
      4_227_882_000, "animals", "lower", FAO, "Meat of ducks, World, 2024: 4,227,882 (1000 An)", tol=0.2),
    f("fao_turkeys", "According to FAOSTAT, roughly how many turkeys were slaughtered worldwide in 2024? Give the number of animals.",
      504_128_000, "animals", "lower", FAO, "Meat of turkeys, World, 2024: 504,128 (1000 An)", tol=0.2),
]

PROMPT = ("{q}\nGive your best single estimate even if unsure. Reply with only a JSON object: "
          '{{"answer": <number only, in {unit}>, "confidence": <0-100, your confidence that this is correct>}}')
