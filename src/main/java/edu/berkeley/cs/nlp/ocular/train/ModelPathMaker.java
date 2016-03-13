package edu.berkeley.cs.nlp.ocular.train;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class ModelPathMaker {

	public static String makeGsmPath(String outputPath, int iter, int batch, String suffix) {
		return outputPath + "/gsm/" + makeOutputFilePrefix(iter, batch) + suffix + ".gsmser";
	}

	public static String makeLmPath(String outputPath, int iter, int batch) {
		return outputPath + "/lm/" + makeOutputFilePrefix(iter, batch) + ".lmser";
	}

	public static String makeFontPath(String outputPath, int iter, int batch) {
		return outputPath + "/font/" + makeOutputFilePrefix(iter, batch) + ".fontser";
	}

	private static String makeOutputFilePrefix(int iter, int batch) {
		return "retrained_iter-"+iter+"_batch-"+batch;
	}

}
