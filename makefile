all:
	echo $a
	echo "Mirá el archivo antes de usarlo"

mirror:
	sh git-mirror.sh git@github.com:BayesDeLasProvinciasUnidasDelSur/variational-logisitic-regression.git
	sh git-mirror.sh git@git.exactas.uba.ar:bayes/variational-logisitic-regression.git


