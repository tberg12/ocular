package edu.berkeley.cs.nlp.ocular.main;

import java.util.Set;

import edu.berkeley.cs.nlp.ocular.gsm.BasicGlyphSubstitutionModel.BasicGlyphSubstitutionModelFactory;
import edu.berkeley.cs.nlp.ocular.gsm.GlyphSubstitutionModel;
import edu.berkeley.cs.nlp.ocular.gsm.GlyphSubstitutionModelReadWrite;
import edu.berkeley.cs.nlp.ocular.lm.CodeSwitchLanguageModel;
import fig.Option;
import fig.OptionsParser;
import indexer.Indexer;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class InitializeGlyphSubstitutionModel implements Runnable {
	
	@Option(gloss = "Path to the language model file (so that it knows which characters to create images for).")
	public static String inputLmPath = null; // Required.

	@Option(gloss = "Output font file path.")
	public static String outputGsmPath = null; // Required.
	
	@Option(gloss = "The default number of counts that every glyph gets in order to smooth the glyph substitution model estimation.")
	public static double gsmSmoothingCount = 1.0;
	
	@Option(gloss = "gsmElisionSmoothingCountMultiplier.")
	public static double gsmElisionSmoothingCountMultiplier = 100.0;
	
	@Option(gloss = "Exponent on GSM scores.")
	public static double gsmPower = 4.0;

	public static void main(String[] args) {
		InitializeGlyphSubstitutionModel main = new InitializeGlyphSubstitutionModel();
		OptionsParser parser = new OptionsParser();
		parser.doRegisterAll(new Object[] {main});
		if (!parser.doParse(args)) System.exit(1);
		main.run();
	}

	public void run() {
		if (inputLmPath == null) throw new IllegalArgumentException("-lmPath not set");
		if (outputGsmPath == null) throw new IllegalArgumentException("-fontPath not set");

		final CodeSwitchLanguageModel lm = InitializeLanguageModel.readCodeSwitchLM(inputLmPath);
		final Indexer<String> charIndexer = lm.getCharacterIndexer();
		final Indexer<String> langIndexer = lm.getLanguageIndexer();
		Set<Integer>[] activeCharacterSets = FonttrainTranscribeShared.makeActiveCharacterSets(lm);
		
		// Fake stuff
		int minCountsForEvalGsm = 0;
		String outputPath = null;
		
		BasicGlyphSubstitutionModelFactory factory = new BasicGlyphSubstitutionModelFactory(
				gsmSmoothingCount, gsmElisionSmoothingCountMultiplier, 
				langIndexer, charIndexer, 
				activeCharacterSets, gsmPower, minCountsForEvalGsm, outputPath);
		
		GlyphSubstitutionModel gsm = factory.uniform();
		
		GlyphSubstitutionModelReadWrite.writeGSM(gsm, outputGsmPath);
	}
	
}
