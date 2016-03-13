package edu.berkeley.cs.nlp.ocular.train;

import static edu.berkeley.cs.nlp.ocular.train.ModelPathMaker.makeFontPath;
import static edu.berkeley.cs.nlp.ocular.train.ModelPathMaker.makeGsmPath;
import static edu.berkeley.cs.nlp.ocular.train.ModelPathMaker.makeLmPath;
import static edu.berkeley.cs.nlp.ocular.util.Tuple2.Tuple2;
import static edu.berkeley.cs.nlp.ocular.util.Tuple3.Tuple3;

import java.io.File;

import edu.berkeley.cs.nlp.ocular.font.Font;
import edu.berkeley.cs.nlp.ocular.lm.CodeSwitchLanguageModel;
import edu.berkeley.cs.nlp.ocular.main.InitializeFont;
import edu.berkeley.cs.nlp.ocular.main.InitializeLanguageModel;
import edu.berkeley.cs.nlp.ocular.sub.GlyphSubstitutionModel;
import edu.berkeley.cs.nlp.ocular.sub.GlyphSubstitutionModelReadWrite;
import edu.berkeley.cs.nlp.ocular.util.Tuple2;
import edu.berkeley.cs.nlp.ocular.util.Tuple3;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class TrainingRestarter {

	/**
	 * If requested, try and pick up where we left off
	 */
	public Tuple2<Tuple2<Integer,Integer>, Tuple3<Font, CodeSwitchLanguageModel, GlyphSubstitutionModel>> getRestartModels(
			String outputPath, int numUsableDocs, int minDocBatchSize, int updateDocBatchSize,
			CodeSwitchLanguageModel lm, GlyphSubstitutionModel gsm, Font font, boolean retrainLM, boolean trainGsm, boolean evalGsmExists,
			int numEMIters) {

		Font newFont = font;
		CodeSwitchLanguageModel newLm = lm;
		GlyphSubstitutionModel newGsm = gsm;
		
		int lastCompletedIteration = 0;
		int lastCompletedBatchOfIteration = 0;
		String fontPath = null;
		int lastBatchNumOfIteration = getLastBatchNumOfIteration(numUsableDocs, updateDocBatchSize, minDocBatchSize);
		for (int iter = 1; iter <= numEMIters; ++iter) {
			fontPath = makeFontPath(outputPath, iter, lastBatchNumOfIteration);
			if (new File(fontPath).exists()) {
				lastCompletedIteration = iter;
			}
		}
		if (lastCompletedIteration > 0) {
			System.out.println("Last completed iteration: "+lastCompletedIteration);
			if (fontPath != null) {
				String lastFontPath = makeFontPath(outputPath, lastCompletedIteration, lastBatchNumOfIteration);
				System.out.println("    Loading font of last completed iteration: "+lastFontPath);
				newFont = InitializeFont.readFont(lastFontPath);
			}
			if (retrainLM) {
				String lastLmPath = makeLmPath(outputPath, lastCompletedIteration, lastBatchNumOfIteration);
				System.out.println("    Loading gsm of last completed iteration:  "+lastLmPath);
				newLm = InitializeLanguageModel.readLM(lastLmPath);
			}
			if (trainGsm) {
				String lastGsmPath = makeGsmPath(outputPath, lastCompletedIteration, lastBatchNumOfIteration, "");
				System.out.println("    Loading lm of last completed iteration:   "+lastGsmPath);
				if (evalGsmExists) newGsm = GlyphSubstitutionModelReadWrite.readGSM(lastGsmPath);
			}
		}
		else {
			System.out.println("No completed iterations found");
		}
		
		if (lastCompletedIteration == numEMIters) {
			System.out.println("All iterations are already complete!");
		}
		
		return Tuple2(Tuple2(lastCompletedIteration, lastCompletedBatchOfIteration), Tuple3(newFont,newLm,newGsm));
	}

	private int getLastBatchNumOfIteration(int numUsableDocs, int updateDocBatchSize, int minDocBatchSize) {
		int completedBatchesInIteration = 0;
		for (int docNum = 0; docNum < numUsableDocs; ++docNum) {
			if (FontTrainer.isBatchComplete(numUsableDocs, docNum, updateDocBatchSize, minDocBatchSize)) {
				++completedBatchesInIteration;
			}
		}
		return completedBatchesInIteration;
	}

}
