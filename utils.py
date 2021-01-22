import threading


def set_interval(func, sec):
    def func_wrapper():
        set_interval(func, sec)
        func()
    t = threading.Timer(sec, func_wrapper)
    t.start()
    return t


__cards = {
    1: '1️⃣',
    2: '2️⃣',
    3: '3️⃣',
    4: '4️⃣',
    5: '5️⃣',
    6: '6️⃣',
    7: '7️⃣',
    8: '8️⃣',
    9: '9️⃣',
    10: '🔟',
    11: "J",
    12: "Q",
    13: "K",
    14: "A",
}


def get_card_by_id(id: float) -> str:
    i = int(id)
    return 'UNKNOWN' if i not in __cards else __cards[i]
