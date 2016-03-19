package edu.berkeley.cs.nlp.ocular.train;

import static edu.berkeley.cs.nlp.ocular.train.ModelPathMaker.makeFontPath;
import static edu.berkeley.cs.nlp.ocular.train.ModelPathMaker.makeGsmPath;
import static edu.berkeley.cs.nlp.ocular.train.ModelPathMaker.makeLmPath;
import static edu.berkeley.cs.nlp.ocular.util.Tuple2.Tuple2;
import static edu.berkeley.cs.nlp.ocular.util.Tuple3.Tuple3;

import java.io.File;

import edu.berkeley.cs.nlp.ocular.font.Font;
import edu.berkeley.cs.nlp.ocular.gsm.GlyphSubstitutionModel;
import edu.berkeley.cs.nlp.ocular.lm.CodeSwitchLanguageModel;
import edu.berkeley.cs.nlp.ocular.main.InitializeFont;
import edu.berkeley.cs.nlp.ocular.main.InitializeGlyphSubstitutionModel;
import edu.berkeley.cs.nlp.ocular.main.InitializeLanguageModel;
import edu.berkeley.cs.nlp.ocular.util.Tuple2;
import edu.berkeley.cs.nlp.ocular.util.Tuple3;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class TrainingRestarter {

	/**
	 * If requested, try and pick up where we left off
	 */
	public Tuple2<Integer, Tuple3<Font, CodeSwitchLanguageModel, GlyphSubstitutionModel>> getRestartModels(
			Font inputFont, CodeSwitchLanguageModel inputLm, GlyphSubstitutionModel inputGsm, 
			boolean updateLM, boolean updateGsm, String outputPath,
			int numEMIters, int numUsableDocs, int updateDocBatchSize, boolean noUpdateIfBatchTooSmall) {

		int lastCompletedIteration = 0;
		String fontPath = null;
		int lastBatchNumOfIteration = getLastBatchNumOfIteration(numUsableDocs, updateDocBatchSize, noUpdateIfBatchTooSmall);
		for (int iter = 1; iter <= numEMIters; ++iter) {
			fontPath = makeFontPath(outputPath, iter, lastBatchNumOfIteration);
			if (new File(fontPath).exists()) {
				lastCompletedIteration = iter;
			}
		}
		
		Font newFont = inputFont;
		CodeSwitchLanguageModel newLm = inputLm;
		GlyphSubstitutionModel newGsm = inputGsm;
		
		if (lastCompletedIteration == numEMIters) {
			System.out.println("All iterations are already complete!");
		}
		else if (lastCompletedIteration > 0) {
			System.out.println("Last completed iteration: "+lastCompletedIteration);
			if (fontPath != null) {
				String lastFontPath = makeFontPath(outputPath, lastCompletedIteration, lastBatchNumOfIteration);
				System.out.println("    Loading font of last completed iteration: "+lastFontPath);
				newFont = InitializeFont.readFont(lastFontPath);
			}
			if (updateLM) {
				String lastLmPath = makeLmPath(outputPath, lastCompletedIteration, lastBatchNumOfIteration);
				System.out.println("    Loading lm of last completed iteration:  "+lastLmPath);
				newLm = InitializeLanguageModel.readCodeSwitchLM(lastLmPath);
			}
			if (updateGsm) {
				String lastGsmPath = makeGsmPath(outputPath, lastCompletedIteration, lastBatchNumOfIteration);
				System.out.println("    Loading gsm of last completed iteration:   "+lastGsmPath);
				newGsm = InitializeGlyphSubstitutionModel.readGSM(lastGsmPath);
			}
		}
		else {
			System.out.println("No completed iterations found");
		}
		
		return Tuple2(lastCompletedIteration, Tuple3(newFont,newLm,newGsm));
	}

	private int getLastBatchNumOfIteration(int numUsableDocs, int updateDocBatchSize, boolean noUpdateIfBatchTooSmall) {
		int completedBatchesInIteration = 0;
		int currentBatchSize = 0;
		for (int docNum = 0; docNum < numUsableDocs; ++docNum) {
			++currentBatchSize;
			if (FontTrainer.isBatchComplete(numUsableDocs, docNum, currentBatchSize, updateDocBatchSize, noUpdateIfBatchTooSmall)) {
				++completedBatchesInIteration;
			}
		}
		return completedBatchesInIteration;
	}

}
