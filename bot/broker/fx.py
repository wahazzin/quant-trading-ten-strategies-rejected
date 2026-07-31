from ib_async import Forex


class FXConverter:
    """
    Fetches live FX rates from IBKR and converts amounts between currencies.
    """

    def __init__(self, ib):
        self.ib = ib
        self._cache = {}

    def _fetch_rate(self, pair):
        """
        pair example: 'USDSEK' means: how many SEK per 1 USD.
        """
        if pair in self._cache:
            return self._cache[pair]

        contract = Forex(pair)
        self.ib.qualifyContracts(contract)

        ticker = self.ib.reqMktData(contract, "", False, False)
        self.ib.sleep(2)

        rate = ticker.last or ticker.close
        self.ib.cancelMktData(contract)

        if not rate or rate <= 0:
            raise ValueError(f"Could not fetch FX rate for pair {pair}")

        self._cache[pair] = rate
        return rate

    def convert(self, amount, from_currency, to_currency):
        """
        Converts `amount` from `from_currency` into `to_currency`.
        """
        if from_currency == to_currency:
            return amount

        pair = f"{to_currency}{from_currency}"

        try:
            rate = self._fetch_rate(pair)
            # rate = how many `from_currency` units per 1 `to_currency` unit
            return amount / rate
        except Exception:
            reverse_pair = f"{from_currency}{to_currency}"
            rate = self._fetch_rate(reverse_pair)
            return amount * rate