package edu.berkeley.cs.nlp.ocular.main;

import java.io.File;
import java.util.List;
import java.util.Map;

import edu.berkeley.cs.nlp.ocular.font.Font;
import edu.berkeley.cs.nlp.ocular.model.CharacterTemplate;
import tberg.murphy.arrays.a;
import tberg.murphy.fig.Option;
import tberg.murphy.fileio.f;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 */
public class ExportFont extends OcularRunnable {

	@Option(gloss = "Input font file path.")
	public static String inputFontPath = null; // Required.
	
	@Option(gloss = "Input font file path.")
	public static String outputDirPath = null; // Required.
	
	public static void main(String[] args) {
		System.out.println("ExportFont");
		ExportFont main = new ExportFont();
		main.doMain(main, args);
	}
	
	protected void validateOptions() {
		if (inputFontPath == null) throw new IllegalArgumentException("-inputFontPath not set");
		if (outputDirPath == null) throw new IllegalArgumentException("-outputDirPath not set");
	}

	public void run(List<String> commandLineArgs) {
		Font font = InitializeFont.readFont(inputFontPath);
		
		File outDir = new File(outputDirPath);
		outDir.mkdirs();
		
		for (Map.Entry<String,CharacterTemplate> entry : font.charTemplates.entrySet()) {
			CharacterTemplate template = entry.getValue();
			int bestWidth = -1;
			double widthProb = Double.NEGATIVE_INFINITY;
			for (int w=template.templateMinWidth(); w<=template.templateMaxWidth(); ++w) {
				if (widthProb <= template.widthLogProb(w)) {
					bestWidth = w;
					widthProb = template.widthLogProb(w);
				}
			}
			float[][] probs = template.blackProbs(0, 0, bestWidth);
			String charName = entry.getKey();
			if (!charName.matches("[a-zA-Z0-9]+")) charName = Integer.toString((int) charName.charAt(0));
			f.writeString(outputDirPath+"/char_"+entry.getValue(), a.toString(probs));
		}
		
	}

}
