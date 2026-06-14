from bisindo_llm import gloss_to_sentence


cases = [
    ("makan aku suka", "Saya suka makan."),
    ("makan aku tidak_suka", "Saya tidak suka makan."),
    ("makan aku lagi", "Saya sedang makan."),
    ("guru aku bertemu rabu", "Saya bertemu guru pada hari Rabu."),
    ("jumat malam kita bertemu", "Kita bertemu pada Jumat malam."),
    ("sabtu kita bertemu", "Kita bertemu pada hari Sabtu."),
    ("orang_tua aku sehat", "Orang tua saya sehat."),
    ("nama kamu", "Siapa nama kamu?"),
    ("di_mana kita bertemu", "Di mana kita bertemu?"),
    ("kalau kamu sehat kita bertemu", "Kalau kamu sehat, kita bertemu."),
]


for token_text, expected in cases:
    output = gloss_to_sentence(token_text)

    print("=" * 60)
    print("TOKEN    :", token_text)
    print("OUTPUT   :", output)
    print("EXPECTED :", expected)
    print("MATCH    :", output.strip() == expected.strip())
