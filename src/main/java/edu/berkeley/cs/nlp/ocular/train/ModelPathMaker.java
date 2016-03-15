package edu.berkeley.cs.nlp.ocular.train;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class ModelPathMaker {

	public static String makeFontDir(String outputPath) {
		return outputPath + "/font/";
	}
	public static String makeFontPath(String outputPath, int iter, int batch) {
		return makeFontDir(outputPath) + makeOutputFilePrefix(iter, batch) + ".fontser";
	}
	public static String makeFontFilenameRegex() {
		return makeOutputFilePrefixRegex() + ".fontser";
	}
	
	public static String makeLmDir(String outputPath) {
		return outputPath + "/lm/";
	}
	public static String makeLmPath(String outputPath, int iter, int batch) {
		return makeLmDir(outputPath) + makeOutputFilePrefix(iter, batch) + ".lmser";
	}
	public static String makeLmFilenameRegex() {
		return makeOutputFilePrefixRegex() + ".lmser";
	}

	public static String makeGsmDir(String outputPath) {
		return outputPath + "/gsm/";
	}
	public static String makeGsmPath(String outputPath, int iter, int batch) {
		return makeGsmDir(outputPath) + makeOutputFilePrefix(iter, batch) + ".gsmser";
	}
	public static String makeGsmFilenameRegex() {
		return makeOutputFilePrefixRegex() + ".gsmser";
	}
	
	private static String makeOutputFilePrefix(int iter, int batch) {
		return "retrained_iter-"+iter+"_batch-"+batch;
	}
	public static String makeOutputFilePrefixRegex() {
		return "retrained_iter-(\\d+)_batch-(\\d+)";
	}

}
