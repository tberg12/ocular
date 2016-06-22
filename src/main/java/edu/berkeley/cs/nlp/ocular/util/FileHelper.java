package edu.berkeley.cs.nlp.ocular.util;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class FileHelper {

	public static void writeString(String path, String str) {
		BufferedWriter out = null;
		try {
			File f = new File(path);
			f.getAbsoluteFile().getParentFile().mkdirs();
			out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(f), "utf-8"));
			out.write(str);
		} catch (IOException ex) {
			throw new RuntimeException(ex);
		} finally {
			if (out != null) {
				try { out.close(); } catch (Exception ex) {}
			}
		}
	}

}
