package edu.berkeley.cs.nlp.ocular.main;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.List;
import java.util.Set;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import edu.berkeley.cs.nlp.ocular.gsm.BasicGlyphSubstitutionModel.BasicGlyphSubstitutionModelFactory;
import edu.berkeley.cs.nlp.ocular.gsm.GlyphSubstitutionModel;
import edu.berkeley.cs.nlp.ocular.lm.CodeSwitchLanguageModel;
import tberg.murphy.fig.Option;
import tberg.murphy.indexer.Indexer;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class InitializeGlyphSubstitutionModel extends OcularRunnable {
	
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
		System.out.println("InitializeGlyphSubstitutionModel");
		InitializeGlyphSubstitutionModel main = new InitializeGlyphSubstitutionModel();
		main.doMain(main, args);
	}

	protected void validateOptions() {
		if (inputLmPath == null) throw new IllegalArgumentException("-inputLmPath not set");
		if (outputGsmPath == null) throw new IllegalArgumentException("-outputGsmPath not set");
	}

	public void run(List<String> commandLineArgs) {
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
		
		System.out.println("Initializing a uniform GSM.");
		GlyphSubstitutionModel gsm = factory.uniform();
		
		System.out.println("Writing intialized gsm to " + outputGsmPath);
		writeGSM(gsm, outputGsmPath);
	}
	
	public static GlyphSubstitutionModel readGSM(String gsmPath) {
		ObjectInputStream in = null;
		try {
			File file = new File(gsmPath);
			if (!file.exists()) {
				throw new RuntimeException("Serialized GlyphSubstitutionModel file " + gsmPath + " not found");
			}
			in = new ObjectInputStream(new GZIPInputStream(new FileInputStream(file)));
			return (GlyphSubstitutionModel) in.readObject();
		} catch (Exception e) {
			throw new RuntimeException(e);
		} finally {
			if (in != null)
				try { in.close(); } catch (IOException e) { throw new RuntimeException(e); }
		}
	}

	public static void writeGSM(GlyphSubstitutionModel gsm, String gsmPath) {
		ObjectOutputStream out = null;
		try {
			new File(gsmPath).getAbsoluteFile().getParentFile().mkdirs();
			out = new ObjectOutputStream(new GZIPOutputStream(new FileOutputStream(gsmPath)));
			out.writeObject(gsm);
		} catch (Exception e) {
			throw new RuntimeException(e);
		} finally {
			if (out != null)
				try { out.close(); } catch (IOException e) { throw new RuntimeException(e); }
		}
	}

}
