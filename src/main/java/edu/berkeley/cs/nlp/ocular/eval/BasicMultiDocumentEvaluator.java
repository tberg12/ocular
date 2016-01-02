package edu.berkeley.cs.nlp.ocular.eval;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import edu.berkeley.cs.nlp.ocular.data.FileUtil;
import edu.berkeley.cs.nlp.ocular.data.ImageLoader.Document;
import edu.berkeley.cs.nlp.ocular.eval.Evaluator.EvalSuffStats;
import edu.berkeley.cs.nlp.ocular.lm.CodeSwitchLanguageModel;
import edu.berkeley.cs.nlp.ocular.model.CharacterTemplate;
import edu.berkeley.cs.nlp.ocular.model.DecoderEM;
import edu.berkeley.cs.nlp.ocular.model.DenseBigramTransitionModel;
import edu.berkeley.cs.nlp.ocular.model.FontTrainEM;
import edu.berkeley.cs.nlp.ocular.model.SparseTransitionModel.TransitionState;
import edu.berkeley.cs.nlp.ocular.sub.GlyphSubstitutionModel;
import edu.berkeley.cs.nlp.ocular.util.Tuple2;
import indexer.Indexer;

/**
 * @author Dan Garrette (dhg@cs.utexas.edu)
 */
public class BasicMultiDocumentEvaluator implements MultiDocumentEvaluator {
	private List<Document> documents;
	private String inputPath; 
	private String outputPath;
	private DecoderEM decoderEM;
	private SingleDocumentEvaluator docEvaluator;
	private Indexer<String> charIndexer;
	
	public BasicMultiDocumentEvaluator(
			List<Document> documents, String inputPath, String outputPath,
			DecoderEM decoderEM,
			SingleDocumentEvaluator docEvaluator,
			Indexer<String> charIndexer) {
		this.documents = documents;
		this.inputPath = inputPath;
		this.outputPath = outputPath;
		this.decoderEM = decoderEM;
		this.docEvaluator = docEvaluator;
		this.charIndexer = charIndexer;
	}

	public void printTranscriptionWithEvaluation(int iter, int batchId,
			CodeSwitchLanguageModel lm, GlyphSubstitutionModel gsm, Map<String, CharacterTemplate> font) {
		
		int numDocs = documents.size();
		CharacterTemplate[] templates = FontTrainEM.loadTemplates(font, charIndexer);
		DenseBigramTransitionModel backwardTransitionModel = new DenseBigramTransitionModel(lm);
		
		double totalJointLogProb = 0.0;
		List<Tuple2<String, Map<String, EvalSuffStats>>> allEvals = new ArrayList<Tuple2<String, Map<String, EvalSuffStats>>>();
		List<Tuple2<String, Map<String, EvalSuffStats>>> allLmEvals = new ArrayList<Tuple2<String, Map<String, EvalSuffStats>>>();
		for (int docNum = 0; docNum < numDocs; ++docNum) {
			Document doc = documents.get(docNum);
			System.out.println((iter > 0 ? "Training iteration "+iter+", " : "") + (batchId > 0 ? "batch "+batchId+", " : "") + "Transcribing eval document "+(docNum+1)+" of "+numDocs+":  "+doc.baseName());
			doc.loadLineText();
			
			Tuple2<Tuple2<TransitionState[][], int[][]>, Double> decodeResults = decoderEM.computeEStep(doc, false, lm, gsm, templates, backwardTransitionModel);
			final TransitionState[][] decodeStates = decodeResults._1._1;
			final int[][] decodeWidths = decodeResults._1._2;
			totalJointLogProb += decodeResults._2;

			docEvaluator.printTranscriptionWithEvaluation(iter, batchId, doc, decodeStates, decodeWidths, inputPath, outputPath, allEvals, allLmEvals);
		}
		double avgLogProb = ((double)totalJointLogProb) / numDocs;
		System.out.println("Iteration "+iter+", batch "+batchId+": eval avg joint log prob: " + avgLogProb);
		if (new File(inputPath).isDirectory()) {
			Document doc = documents.get(0);
			String fileParent = FileUtil.removeCommonPathPrefixOfParents(new File(inputPath), new File(doc.baseName()))._2;
			String preext = "eval";
			String outputFilenameBase = outputPath + "/" + fileParent + "/" + preext;
			if (iter > 0) outputFilenameBase += "_iter-" + iter;
			if (batchId > 0) outputFilenameBase += "_batch-" + batchId;
			FontTrainEM.printEvaluation(allEvals, outputFilenameBase + ".txt");
			FontTrainEM.printEvaluation(allLmEvals, outputFilenameBase + "_lmeval.txt");
		}
	}

}
