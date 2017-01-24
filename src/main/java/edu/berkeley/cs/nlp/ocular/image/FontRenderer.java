package edu.berkeley.cs.nlp.ocular.image;

import edu.berkeley.cs.nlp.ocular.data.textreader.Charset;
import edu.berkeley.cs.nlp.ocular.image.ImageUtils.PixelType;
import tberg.murphy.indexer.Indexer;

import java.awt.Color;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics2D;
import java.awt.GraphicsEnvironment;
import java.awt.RenderingHints;
import java.awt.geom.AffineTransform;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.unescapeChar;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 */
public class FontRenderer {
	
	private static Set<String> OUTLAWED_FONTS;
	static {
		OUTLAWED_FONTS = new HashSet<String>();
		OUTLAWED_FONTS.add("Vemana2000");
		OUTLAWED_FONTS.add("utkal");
		OUTLAWED_FONTS.add("Untitled1");
		OUTLAWED_FONTS.add("Symbol");
		OUTLAWED_FONTS.add("Standard Symbols L");
		OUTLAWED_FONTS.add("Saab");
		OUTLAWED_FONTS.add("Pothana2000");
		OUTLAWED_FONTS.add("OpenSymbol");
		OUTLAWED_FONTS.add("mry_KacstQurn");
		OUTLAWED_FONTS.add("Mallige");
		OUTLAWED_FONTS.add("Lohit Tamil");
		OUTLAWED_FONTS.add("Lohit Punjabi");
		OUTLAWED_FONTS.add("Lohit Hindi");
		OUTLAWED_FONTS.add("Lohit Gujarati");
		OUTLAWED_FONTS.add("Lohit Bengali");
		OUTLAWED_FONTS.add("Kedage");
		OUTLAWED_FONTS.add("KacstTitleL");
		OUTLAWED_FONTS.add("KacstTitle");
		OUTLAWED_FONTS.add("KacstScreen");
		OUTLAWED_FONTS.add("KacstQurn");
		OUTLAWED_FONTS.add("KacstPoster");
		OUTLAWED_FONTS.add("KacstPen");
		OUTLAWED_FONTS.add("KacstOne");
		OUTLAWED_FONTS.add("KacstOffice");
		OUTLAWED_FONTS.add("KacstNaskh");
		OUTLAWED_FONTS.add("KacstLetter");
		OUTLAWED_FONTS.add("KacstFarsi");
		OUTLAWED_FONTS.add("KacstDigital");
		OUTLAWED_FONTS.add("KacstDecorative");
		OUTLAWED_FONTS.add("KacstBook");
		OUTLAWED_FONTS.add("KacstArt");
		OUTLAWED_FONTS.add("GFS Solomos");
		OUTLAWED_FONTS.add("GFS Porson");
		OUTLAWED_FONTS.add("GFS Olga");
		OUTLAWED_FONTS.add("GFS Gazis");
		OUTLAWED_FONTS.add("GFS Didot Classic");
		OUTLAWED_FONTS.add("GFS Baskerville");
		OUTLAWED_FONTS.add("Dingbats");
		OUTLAWED_FONTS.add("GFS BodoniClassic");
		OUTLAWED_FONTS.add("Webdings");
		OUTLAWED_FONTS.add("Te X Gyre Chorus");
		OUTLAWED_FONTS.add("URW Chancery L");
		OUTLAWED_FONTS.add("TeXGyreChorus");
		OUTLAWED_FONTS.add("Lohit Devanagari");
		OUTLAWED_FONTS.add("Droid Sans Thai");
		OUTLAWED_FONTS.add("Droid Sans Hebrew");
		OUTLAWED_FONTS.add("Droid Sans Georgian");
		OUTLAWED_FONTS.add("Droid Sans Ethiopic");
		OUTLAWED_FONTS.add("Droid Sans Armenian");
		OUTLAWED_FONTS.add("Droid Arabic Naskh");
		OUTLAWED_FONTS.add("STIXIntegralsD");
		OUTLAWED_FONTS.add("STIXIntegralsSm");
		OUTLAWED_FONTS.add("STIXIntegralsUp");
		OUTLAWED_FONTS.add("STIXIntegralsUpD");
		OUTLAWED_FONTS.add("STIXIntegralsUpSm");
		OUTLAWED_FONTS.add("STIXNonUnicode");
		OUTLAWED_FONTS.add("STIXSizeFiveSym");
		OUTLAWED_FONTS.add("STIXSizeFourSym");
		OUTLAWED_FONTS.add("STIXSizeOneSym");
		OUTLAWED_FONTS.add("STIXSizeThreeSym");
		OUTLAWED_FONTS.add("STIXSizeTwoSym");
		OUTLAWED_FONTS.add("STIXVariants");
		OUTLAWED_FONTS.add("TakaoPGothic");
		OUTLAWED_FONTS.add("Droid Sans Japanese");
		OUTLAWED_FONTS.add("LKLUG");
		OUTLAWED_FONTS.add("Tibetan Machine Uni");
		OUTLAWED_FONTS.add("esint10");
		OUTLAWED_FONTS.add("eufm10");
		OUTLAWED_FONTS.add("cmex10");
		OUTLAWED_FONTS.add("cmsy10");
		OUTLAWED_FONTS.add("cmr10");
		OUTLAWED_FONTS.add("Rachana");
		OUTLAWED_FONTS.add("rsfs10");
		OUTLAWED_FONTS.add("wasy10");
		
		OUTLAWED_FONTS.add("Academy Engraved LET");
		OUTLAWED_FONTS.add("Bodoni Ornaments ITC TT");
		OUTLAWED_FONTS.add("Bordeaux Roman Bold LET");
		OUTLAWED_FONTS.add("Braggadocio");
		OUTLAWED_FONTS.add("Curlz MT");
		OUTLAWED_FONTS.add("Didot");
		OUTLAWED_FONTS.add("Desdemona");
		OUTLAWED_FONTS.add("Engravers MT");
		OUTLAWED_FONTS.add("Princetown LET");
		OUTLAWED_FONTS.add("Type Embellishments One LET");
		OUTLAWED_FONTS.add("Wide Latin");
		OUTLAWED_FONTS.add("Wingdings");
		OUTLAWED_FONTS.add("Wingdings 2");
		OUTLAWED_FONTS.add("Wingdings 3");
		OUTLAWED_FONTS.add("Zapf Dingbats");
		OUTLAWED_FONTS.add("Zapfino");

		OUTLAWED_FONTS.add("Al Tarikh");
		OUTLAWED_FONTS.add("Apple Chancery");
		OUTLAWED_FONTS.add("Apple LiGothic");
		OUTLAWED_FONTS.add("Apple LiSung");
		OUTLAWED_FONTS.add("Apple Symbols");
		OUTLAWED_FONTS.add("AppleMyungjo");
		OUTLAWED_FONTS.add("Avenir Next");
		OUTLAWED_FONTS.add("Avenir Next Condensed");
		OUTLAWED_FONTS.add("Ayuthaya");
		OUTLAWED_FONTS.add("Bank Gothic");
		OUTLAWED_FONTS.add("Baoli SC");
		OUTLAWED_FONTS.add("Baskerville Old Face");
		OUTLAWED_FONTS.add("Batang");
		OUTLAWED_FONTS.add("Bauhaus 93");
		OUTLAWED_FONTS.add("Bell MT");
		OUTLAWED_FONTS.add("Bernard MT Condensed");
		OUTLAWED_FONTS.add("BiauKai");
		OUTLAWED_FONTS.add("Bodoni SvtyTwo ITC TT");
		OUTLAWED_FONTS.add("Bodoni SvtyTwo OS ITC TT");
		OUTLAWED_FONTS.add("Bodoni SvtyTwo SC ITC TT");
		OUTLAWED_FONTS.add("Book Antiqua");
		OUTLAWED_FONTS.add("Bookman Old Style");
		OUTLAWED_FONTS.add("Bookshelf Symbol 7");
		OUTLAWED_FONTS.add("Britannic Bold");
		OUTLAWED_FONTS.add("Brush Script MT");
		OUTLAWED_FONTS.add("Calisto MT");
		OUTLAWED_FONTS.add("Cambria");
		OUTLAWED_FONTS.add("Cambria Math");
		OUTLAWED_FONTS.add("Capitals");
		OUTLAWED_FONTS.add("Century");
		OUTLAWED_FONTS.add("Century Gothic");
		OUTLAWED_FONTS.add("Century Schoolbook");
		OUTLAWED_FONTS.add("Chalkboard");
		OUTLAWED_FONTS.add("Chalkboard SE");
		OUTLAWED_FONTS.add("Chalkduster");
		OUTLAWED_FONTS.add("Cochin");
		OUTLAWED_FONTS.add("Colonna MT");
		OUTLAWED_FONTS.add("Comic Sans MS");
		OUTLAWED_FONTS.add("Consolas");
		OUTLAWED_FONTS.add("Constantia");
		OUTLAWED_FONTS.add("Cooper Black");
		OUTLAWED_FONTS.add("Copperplate");
		OUTLAWED_FONTS.add("Copperplate Gothic Bold");
		OUTLAWED_FONTS.add("Copperplate Gothic Light");
		OUTLAWED_FONTS.add("Corsiva Hebrew");
		OUTLAWED_FONTS.add("Courier New");
		OUTLAWED_FONTS.add("Damascus");
		OUTLAWED_FONTS.add("Devanagari MT");
		OUTLAWED_FONTS.add("DIN Alternate");
		OUTLAWED_FONTS.add("DIN Condensed");
		OUTLAWED_FONTS.add("Edwardian Script ITC");
		OUTLAWED_FONTS.add("Eurostile");
		OUTLAWED_FONTS.add("Footlight MT Light");
		OUTLAWED_FONTS.add("Garamond");
		OUTLAWED_FONTS.add("Gloucester MT Extra Condensed");
		OUTLAWED_FONTS.add("Goudy Old Style");
		OUTLAWED_FONTS.add("Gujarati MT");
		OUTLAWED_FONTS.add("Gulim");
		OUTLAWED_FONTS.add("GungSeo");
		OUTLAWED_FONTS.add("Gurmukhi MN");
		OUTLAWED_FONTS.add("Haettenschweiler");
		OUTLAWED_FONTS.add("Hannotate SC");
		OUTLAWED_FONTS.add("Hannotate TC");
		OUTLAWED_FONTS.add("HanziPen SC");
		OUTLAWED_FONTS.add("HanziPen TC");
		OUTLAWED_FONTS.add("Harrington");
		OUTLAWED_FONTS.add("HeadLineA");
		OUTLAWED_FONTS.add("Heiti SC");
		OUTLAWED_FONTS.add("Heiti TC");
		OUTLAWED_FONTS.add("Helvetica CY");
		OUTLAWED_FONTS.add("Helvetica Neue");
		OUTLAWED_FONTS.add("Herculanum");
		OUTLAWED_FONTS.add("Hiragino Kaku Gothic Pro");
		OUTLAWED_FONTS.add("Hiragino Kaku Gothic ProN");
		OUTLAWED_FONTS.add("Hiragino Kaku Gothic Std");
		OUTLAWED_FONTS.add("Hiragino Kaku Gothic StdN");
		OUTLAWED_FONTS.add("Hiragino Maru Gothic Pro");
		OUTLAWED_FONTS.add("Hiragino Maru Gothic ProN");
		OUTLAWED_FONTS.add("Hiragino Mincho Pro");
		OUTLAWED_FONTS.add("Hiragino Mincho ProN");
		OUTLAWED_FONTS.add("Hoefler Text");
		OUTLAWED_FONTS.add("Impact");
		OUTLAWED_FONTS.add("Imprint MT Shadow");
		OUTLAWED_FONTS.add("InaiMathi");
		OUTLAWED_FONTS.add("Jazz LET");
		OUTLAWED_FONTS.add("Kai");
		OUTLAWED_FONTS.add("Kaiti SC");
		OUTLAWED_FONTS.add("Kaiti TC");
		OUTLAWED_FONTS.add("Kannada MN");
		OUTLAWED_FONTS.add("Khmer MN");
		OUTLAWED_FONTS.add("Kino MT");
		OUTLAWED_FONTS.add("Kokonor");
		OUTLAWED_FONTS.add("Krungthep");
		OUTLAWED_FONTS.add("Lao MN");
		OUTLAWED_FONTS.add("Libian SC");
		OUTLAWED_FONTS.add("LiSong Pro");
		OUTLAWED_FONTS.add("Lucida Blackletter");
		OUTLAWED_FONTS.add("Lucida Bright");
		OUTLAWED_FONTS.add("Lucida Calligraphy");
		OUTLAWED_FONTS.add("Lucida Fax");
		OUTLAWED_FONTS.add("Lucida Handwriting");
		OUTLAWED_FONTS.add("Lucida Sans Typewriter");
		OUTLAWED_FONTS.add("Malayalam MN");
		OUTLAWED_FONTS.add("Marion");
		OUTLAWED_FONTS.add("Marlett");
		OUTLAWED_FONTS.add("Matura MT Script Capitals");
		OUTLAWED_FONTS.add("Meiryo");
		OUTLAWED_FONTS.add("Menlo");
		OUTLAWED_FONTS.add("Microsoft Himalaya");
		OUTLAWED_FONTS.add("Microsoft Yi Baiti");
		OUTLAWED_FONTS.add("MingLiU");
		OUTLAWED_FONTS.add("MingLiU-ExtB");
		OUTLAWED_FONTS.add("MingLiU_HKSCS");
		OUTLAWED_FONTS.add("MingLiU_HKSCS-ExtB");
		OUTLAWED_FONTS.add("Mistral");
		OUTLAWED_FONTS.add("Modern No. 20");
		OUTLAWED_FONTS.add("Mona Lisa Solid ITC TT");
		OUTLAWED_FONTS.add("Mongolian Baiti");
		OUTLAWED_FONTS.add("Monospaced");
		OUTLAWED_FONTS.add("Monotype Corsiva");
		OUTLAWED_FONTS.add("MS Gothic");
		OUTLAWED_FONTS.add("MS Mincho");
		OUTLAWED_FONTS.add("MS PGothic");
		OUTLAWED_FONTS.add("MS PMincho");
		OUTLAWED_FONTS.add("MS Reference Sans Serif");
		OUTLAWED_FONTS.add("Mshtakan");
		OUTLAWED_FONTS.add("Myanmar MN");
		OUTLAWED_FONTS.add("Nanum Brush Script");
		OUTLAWED_FONTS.add("Nanum Myeongjo");
		OUTLAWED_FONTS.add("Nanum Pen Script");
		OUTLAWED_FONTS.add("Noteworthy");
		OUTLAWED_FONTS.add("Onyx");
		OUTLAWED_FONTS.add("Optima");
		OUTLAWED_FONTS.add("Oriya MN");
		OUTLAWED_FONTS.add("Oriya Sangam MN");
		OUTLAWED_FONTS.add("Palatino");
		OUTLAWED_FONTS.add("Palatino Linotype");
		OUTLAWED_FONTS.add("Papyrus");
		OUTLAWED_FONTS.add("Party LET");
		OUTLAWED_FONTS.add("PCMyungjo");
		OUTLAWED_FONTS.add("Perpetua");
		OUTLAWED_FONTS.add("Perpetua Titling MT");
		OUTLAWED_FONTS.add("PilGi");
		OUTLAWED_FONTS.add("Plantagenet Cherokee");
		OUTLAWED_FONTS.add("Playbill");
		OUTLAWED_FONTS.add("PMingLiU");
		OUTLAWED_FONTS.add("PMingLiU-ExtB");
		OUTLAWED_FONTS.add("PortagoITC TT");
		OUTLAWED_FONTS.add("PT Serif");
		OUTLAWED_FONTS.add("PT Serif Caption");
		OUTLAWED_FONTS.add("Rockwell");
		OUTLAWED_FONTS.add("Rockwell Extra Bold");
		OUTLAWED_FONTS.add("Santa Fe LET");
		OUTLAWED_FONTS.add("Savoye LET");
		OUTLAWED_FONTS.add("SchoolHouse Cursive B");
		OUTLAWED_FONTS.add("SchoolHouse Printed A");
		OUTLAWED_FONTS.add("Seravek");
		OUTLAWED_FONTS.add("Serif");
		OUTLAWED_FONTS.add("Sinhala MN");
		OUTLAWED_FONTS.add("Snell Roundhand");
		OUTLAWED_FONTS.add("Songti SC");
		OUTLAWED_FONTS.add("Songti TC");
		OUTLAWED_FONTS.add("Stencil");
		OUTLAWED_FONTS.add("STFangsong");
		OUTLAWED_FONTS.add("STHeiti");
		OUTLAWED_FONTS.add("STKaiti");
		OUTLAWED_FONTS.add("STSong");
		OUTLAWED_FONTS.add("Superclarendon");
		OUTLAWED_FONTS.add("Synchro LET");
		OUTLAWED_FONTS.add("Tahoma");
		OUTLAWED_FONTS.add("Tamil MN");
		OUTLAWED_FONTS.add("Tamil Sangam MN");
		OUTLAWED_FONTS.add("Telugu MN");
		OUTLAWED_FONTS.add("Times");
		OUTLAWED_FONTS.add("Times New Roman");
		OUTLAWED_FONTS.add("Wawati SC");
		OUTLAWED_FONTS.add("Wawati TC");
		OUTLAWED_FONTS.add("Xingkai SC");
		OUTLAWED_FONTS.add("YuGothicYuMincho");
	}

	private static List<String> getFontsToUse(Set<String> allowedFonts) {
		List<String> fontsToUse = new ArrayList<String>();
		GraphicsEnvironment ge = GraphicsEnvironment.getLocalGraphicsEnvironment();
		for (String fontName : ge.getAvailableFontFamilyNames()) {
			if (!allowedFonts.isEmpty()) {
				if (allowedFonts.contains(fontName)) {
					fontsToUse.add(fontName); 
				}
			}
			else if (!OUTLAWED_FONTS.contains(fontName)) {
				fontsToUse.add(fontName);
			}
		}
		return fontsToUse;
	}
	
	public static void main(String[] args) {
		Set<String> allowedFonts = new HashSet<String>();
		for (String fontName : getFontsToUse(allowedFonts)) {
			System.out.println(fontName);
			PixelType[][] data = renderString(fontName, "q̃", 30, "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");
			StringBuffer buf = new StringBuffer();
			if (data.length > 0) {
				for (int j = 0; j < data[0].length; ++j) {
					for (int i = 0; i < data.length; ++i) {
						PixelType val = data[i][j];
						if (val == PixelType.WHITE)
							buf.append(". ");
						else if (val == PixelType.BLACK) 
							buf.append("O ");
						else if (val == PixelType.OBSCURED) 
							buf.append("X ");
					}
					buf.append("\n");
				}
			}
			System.out.println(buf.toString());
		}
	}
	
	public static PixelType[][][][] getRenderedFont(Indexer<String> charIndexer, int height, Set<String> allowedFonts) {
		StringBuffer alphabetStr = new StringBuffer();
		for (int c=0; c<charIndexer.size(); ++c) {
			alphabetStr.append(unescapeChar(charIndexer.getObject(c)));
		}
		
		PixelType[][][][] result = new PixelType[charIndexer.size()][][][];
		for (int c=0; c<charIndexer.size(); ++c) {
			List<PixelType[][]> rendered = new ArrayList<ImageUtils.PixelType[][]>();
			for (String font : getFontsToUse(allowedFonts)) {
				PixelType[][] renderedChar = renderString(font, unescapeChar(charIndexer.getObject(c)), height, alphabetStr.toString());
				if (renderedChar.length > 0) rendered.add(renderedChar);
				else {
					if (!Charset.SPACE.equals(charIndexer.getObject(c))) { 
						System.out.println("Ignoring empty character rendering: " + font +", " + charIndexer.getObject(c));
					}
				}
			}
			result[c] = rendered.toArray(new PixelType[0][][]);
		}
		return result;
	}
	
	private static PixelType[][] renderString(String fontName, String s, int height, String alphabetString) {
		BufferedImage image = new BufferedImage(height, height, BufferedImage.TYPE_BYTE_GRAY);
		Graphics2D g = image.createGraphics();
		g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
		g.setPaint(Color.WHITE);
		g.fillRect(0, 0, height, height);
		
		g.setPaint(Color.BLACK);
		Font font = new Font(fontName, Font.PLAIN, 10);
		g.setFont(font);
		FontMetrics fm = g.getFontMetrics(font);
		fm.getAscent();
		Rectangle2D strBounds = fm.getStringBounds(s, g);
		Rectangle2D alphabetBounds = fm.getStringBounds(alphabetString, g);
		AffineTransform affine = new AffineTransform();
		affine.translate(((double) height)/2.0 - strBounds.getCenterX()*((double) height)/alphabetBounds.getHeight(), ((double) height)/2.0 - alphabetBounds.getCenterY()*((double) height)/alphabetBounds.getHeight());
		affine.scale(((double) height)/alphabetBounds.getHeight(), ((double) height)/(alphabetBounds.getHeight()));
		g.setTransform(affine);
		g.drawString(s, 0, 0);

    	double[][] levels = ImageUtils.getLevels(image);
    	levels = horizontalCrop(levels);
    	return ImageUtils.getPixelTypes(levels);
	}
	
	private static double[][] horizontalCrop(double[][] levels) {
		double whitenessThresh = 0.4 * ImageUtils.MAX_LEVEL;
		int left = -1;
		for (int i=0; i<levels.length; ++i) {
			if (!allAbove(levels[i], whitenessThresh)) {
				left = i;
				break;
			}
		}
		int right = -1;
		for (int i=levels.length-1; i>=0; --i) {
			if (!allAbove(levels[i], whitenessThresh)) {
				right = i+1;
				break;
			}
		}
		if (left == -1) return new double[0][];
		double[][] result = new double[right-left][];
		for (int i=0; i<right-left; ++i) {
			result[i] = levels[left+i];
		}
		return result;
	}
	
	private static boolean allAbove(double[] vect, double threshold) {
		for (double val : vect) {
			if (val < threshold) {
				return false;
			}
		}
		return true;
	}
	
}
