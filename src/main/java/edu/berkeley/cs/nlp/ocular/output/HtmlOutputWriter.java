package edu.berkeley.cs.nlp.ocular.output;

import java.io.File;
import java.util.List;

import edu.berkeley.cs.nlp.ocular.data.textreader.Charset;
import edu.berkeley.cs.nlp.ocular.gsm.GlyphChar;
import edu.berkeley.cs.nlp.ocular.gsm.GlyphChar.GlyphType;
import edu.berkeley.cs.nlp.ocular.model.DecodeState;
import edu.berkeley.cs.nlp.ocular.model.transition.SparseTransitionModel.TransitionState;
import edu.berkeley.cs.nlp.ocular.util.FileUtil;
import tberg.murphy.fileio.f;
import tberg.murphy.indexer.Indexer;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class HtmlOutputWriter {
	
	private Indexer<String> charIndexer;
	private Indexer<String> langIndexer;
	
	public HtmlOutputWriter(Indexer<String> charIndexer, Indexer<String> langIndexer) {
		this.charIndexer = charIndexer;
		this.langIndexer = langIndexer;
	}

	public void write(int numLines, List<DecodeState>[] viterbiTransStates, String imgFilename, String outputFilenameBase) {
		String htmlOutputFilename = outputFilenameBase + ".html";
		
		StringBuffer outputBuffer = new StringBuffer();
		outputBuffer.append("<HTML xmlns=\"http://www.w3.org/1999/xhtml\" xml:lang=\"en\" lang=\"en\">\n");
		outputBuffer.append("<HEAD><META http-equiv=\"Content-Type\" content=\"text/html; charset=UTF-8\"></HEAD>\n");
		outputBuffer.append("<body>\n");
		outputBuffer.append("<table><tr><td>\n");

		String[] colors = new String[] { "Black", "Red", "Blue", "Olive", "Orange", "Magenta", "Lime", "Cyan", "Purple", "Green", "Brown" };

		int prevLanguage = -1;
		for (int line = 0; line < numLines; ++line) {
			for (DecodeState ds : viterbiTransStates[line]) {
				TransitionState ts = ds.ts;
				int lmChar = ts.getLmCharIndex();
				GlyphChar glyph = ts.getGlyphChar();
				int glyphChar = glyph.templateCharIndex;
				String sglyphChar = Charset.unescapeChar(charIndexer.getObject(glyphChar));

				int currLanguage = ts.getLanguageIndex();
				if (currLanguage != prevLanguage) {
					outputBuffer.append("<font color=\"" + colors[currLanguage+1] + "\">");
				}
				
				if (lmChar != glyphChar || glyph.glyphType != GlyphType.NORMAL_CHAR) {
					String norm = Charset.unescapeChar(charIndexer.getObject(lmChar));
					String dipl = (glyph.glyphType == GlyphType.DOUBLED ? "2x"+sglyphChar : glyph.isElided() ? "" : sglyphChar);
					outputBuffer.append("[" + norm + "/" + dipl + "]");
				}
				else {
					outputBuffer.append(sglyphChar);
				}

				prevLanguage = currLanguage;
			}
			outputBuffer.append("</br>\n");
		}
		outputBuffer.append("</font></font><br/><br/><br/>\n");
		for (int i = -1; i < langIndexer.size(); ++i) {
			outputBuffer.append("<font color=\"" + colors[i+1] + "\">" + (i < 0 ? "none" : langIndexer.getObject(i)) + "</font></br>\n");
		}

		outputBuffer.append("</td><td>\n");
		outputBuffer.append("<img src=\"" + FileUtil.pathRelativeTo(imgFilename, new File(htmlOutputFilename).getParent()) + "\" style=\"width: 75%; height: 75%\">\n");
		outputBuffer.append("</td></tr></table>\n");
		outputBuffer.append("</body></html>\n");
		outputBuffer.append("\n\n\n");
		outputBuffer.append("\n\n\n\n\n");
		String outputString = outputBuffer.toString();

		System.out.println("Writing html output to " + htmlOutputFilename);
		f.writeString(htmlOutputFilename, outputString);
	}
	
}
