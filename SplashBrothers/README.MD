#SplashBrothers

Apache-log generator, copyright SnapLogic 2016.  Written and tested with
Python 3, but believed to be Python 2.x compatible.

Run with `-h` or `--help` to see the configuration options.


##Notes on endpoints.json:

This is actually a list of JSONs, not a JSON, but
in Python json.loads() will load appropriately.

To determine frequency that a page or code is called,
all the weights of the code or endpoint are summed, and
then entry-weight / sum(weight) will
be the probability that page or code is called.

Thanks to http://www.useragentstring.com/pages/Browserlist/
for providing an excellent list of obscure (and common)
user-agent strings.
