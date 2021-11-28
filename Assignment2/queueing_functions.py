import numpy

class MethodNotSupportedError(Exception):
    def __init__(self, method:str) -> None:
        self.method = method
        self.message = f"Queueing method {method} is not supported"
        super().__init__(self.message)

def customers_in_system(lmd, mu, method="MM1"):
    """Calculates average number of customers in the system (L_s)

        Args:
            lmd (float): mean arrival rate
            mu (float): mean service rate per busy server
            method (str, optional): type of distribution. Defaults to "MM1".

        Returns:
            float: average time a costumer will spend in the system

        >>> customers_in_system(6, 10)
        1.5
        >>> customers_in_system(6, 10, "MD1")
        1.05
    """
    if method == "MM1": return lmd / (mu - lmd)
    if method == "MD1": return customers_in_queue(lmd,mu, method) + lmd / mu
    raise MethodNotSupportedError(method)

def customers_in_queue(lmd, mu, method="MM1"):
    """Calculates average number of customers in queue/length in queue (L_q)

        Args:
            lmd (float): mean arrival rate
            mu (float): mean service rate per busy server
            method (str, optional): Type of distrubution. Defaults to "MM1".

        Returns:
            float: average number of customers

        >>> customers_in_queue(6,10)
        0.9
        >>> customers_in_queue(6,10, "MD1")
        0.45
    """
    if method == "MM1": return lmd**2 / (mu * (mu - lmd))
    if method == "MD1": return lmd**2 / (2*mu *(mu - lmd))
    raise MethodNotSupportedError(method)

def waiting_time_system(lmd, mu, method="MM1"):
    """ Calculates average waiting time in system (W_s)

        >>> waiting_time_system(6,10)
        0.25
        >>> waiting_time_system(6,10, "MD1")
        0.175

    """

    if method =="MM1": return 1 / (mu - lmd)
    if method =="MD1": return waiting_time_queue(lmd, mu, method) + 1 / mu
    raise MethodNotSupportedError(method)

def waiting_time_queue(lmd, mu, method="MM1"):
    """ Calculates average waiting time in queue (W_q)

        >>> waiting_time_queue(6,10)
        0.15
        >>> waiting_time_queue(6,10, "MD1")
        0.075
    """
    if method =="MM1": return lmd / (mu * (mu - lmd))
    if method =="MD1": return lmd / (2 * mu * (mu-lmd))
    raise MethodNotSupportedError(method)


if __name__ == "__main__":
    import doctest
    doctest.testmod()