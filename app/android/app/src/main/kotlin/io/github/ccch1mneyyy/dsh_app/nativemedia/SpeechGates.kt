package io.github.ccch1mneyyy.dsh_app.nativemedia

import ai.onnxruntime.OrtEnvironment
import android.content.res.AssetManager
import android.util.Log
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.security.DigestInputStream
import java.security.MessageDigest
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * The models ship in the APK or download on demand to [cacheDir] (using
 * `assets/speech/manifest.json` written by the slim APK pipeline); where ONNX
 * Runtime cannot load them, models are not yet downloaded, or once crashed
 * the app on this phone ([NativeTrial]), the energy gate does the work.
 */
object SpeechGates {
    private val downloader = Executors.newSingleThreadExecutor { Thread(it, "dsh-speech-dl") }
    private val downloading = AtomicBoolean(false)

    fun create(
        assets: AssetManager,
        echoCancelled: Boolean,
        marks: NativeTrial.Marks,
        cacheDir: File? = null,
    ): TurnGate {
        var vad: OrtSilero? = null
        var turn: OrtSmartTurn? = null
        var trial: NativeTrial? = null
        try {
            val vadBytes = loadOrSchedule(assets, "silero_vad.onnx", cacheDir)
            val turnBytes = loadOrSchedule(assets, "smart-turn-v3.2-cpu.onnx", cacheDir)
            if (vadBytes == null || turnBytes == null) {
                Log.i("SpeechGates", "Neural speech models downloading in background; using energy gate for this call")
                return SpeechGate(echoCancelled)
            }
            trial = NativeTrial(marks, "ort-" + OrtEnvironment.getEnvironment().version)
            if (trial.blocked()) {
                Log.w("SpeechGates", "ONNX Runtime crashed here before; using energy gate")
                return SpeechGate(echoCancelled)
            }
            val proving = !trial.proven
            if (proving) trial.begin()
            vad = OrtSilero(vadBytes)
            turn = OrtSmartTurn(turnBytes)
            // Initialize the streaming VAD; smart-turn runs on its worker, and
            // once here the first time, so both have answered before the mark goes.
            vad.probability(ShortArray(512))
            vad.reset()
            if (proving) {
                turn.probability(FloatArray(NeuralGate.WINDOW_SAMPLES))
                trial.passed()
            }
            return NeuralGate(vad, turn, echoCancelled, onTurnFailed = {
                Log.w("SpeechGates", "Turn model failed; pauses end after the energy gate's hangover", it)
            })
        } catch (e: Exception) {
            trial?.dropped()
            Log.w("SpeechGates", "Neural models unavailable; using energy gate", e)
        } catch (e: LinkageError) {
            trial?.dropped()
            Log.w("SpeechGates", "ONNX Runtime unavailable; using energy gate", e)
        }
        turn?.close()
        vad?.close()
        return SpeechGate(echoCancelled)
    }

    private fun loadOrSchedule(assets: AssetManager, name: String, cacheDir: File?): ByteArray? {
        runCatching { return loadAsset(assets, name) }
        val dir = cacheDir ?: return null
        val manifest = readManifest(assets) ?: return null
        val entry = manifest.optJSONObject("models")?.optJSONObject(name) ?: return null
        val expectedSha = entry.optString("sha256").ifEmpty { null } ?: return null
        val url = entry.optString("url").ifEmpty { null } ?: return null

        val cached = File(dir, name)
        val shaFile = File(dir, "$name.sha256")
        if (cached.isFile && shaFile.isFile && shaFile.readText().trim().equals(expectedSha, ignoreCase = true)) {
            return runCatching { cached.readBytes() }.getOrNull()
        }
        scheduleDownload(dir, manifest)
        return null
    }

    private fun readManifest(assets: AssetManager): JSONObject? = runCatching {
        assets.open("speech/manifest.json").bufferedReader().use { JSONObject(it.readText()) }
    }.getOrNull()

    private fun scheduleDownload(dir: File, manifest: JSONObject) {
        if (!downloading.compareAndSet(false, true)) return
        downloader.execute {
            try {
                dir.mkdirs()
                val models = manifest.optJSONObject("models") ?: return@execute
                val keys = models.keys()
                while (keys.hasNext()) {
                    val name = keys.next()
                    val entry = models.optJSONObject(name) ?: continue
                    val expectedSha = entry.optString("sha256").ifEmpty { null } ?: continue
                    val url = entry.optString("url").ifEmpty { null } ?: continue
                    val dest = File(dir, name)
                    val shaFile = File(dir, "$name.sha256")
                    if (dest.isFile && shaFile.isFile && shaFile.readText().trim().equals(expectedSha, ignoreCase = true)) {
                        continue
                    }
                    val tmp = File(dir, "$name.part")
                    try {
                        val actual = downloadFile(url, tmp)
                        if (!actual.equals(expectedSha, ignoreCase = true)) {
                            tmp.delete()
                            Log.w("SpeechGates", "Checksum mismatch for $name (got $actual, expected $expectedSha)")
                            continue
                        }
                        if (!tmp.renameTo(dest)) {
                            tmp.copyTo(dest, overwrite = true)
                            tmp.delete()
                        }
                        shaFile.writeText(expectedSha)
                        Log.i("SpeechGates", "Downloaded speech model $name (${dest.length()} bytes)")
                    } catch (e: Exception) {
                        tmp.delete()
                        Log.w("SpeechGates", "Failed to download speech model $name", e)
                    }
                }
            } finally {
                downloading.set(false)
            }
        }
    }

    private fun downloadFile(url: String, dest: File): String {
        dest.delete()
        var current = url
        var redirects = 0
        while (true) {
            val conn = (URL(current).openConnection() as HttpURLConnection).apply {
                instanceFollowRedirects = false
                connectTimeout = 30_000
                readTimeout = 60_000
                setRequestProperty("User-Agent", "dsh-go-android")
            }
            try {
                val code = conn.responseCode
                if (code in listOf(301, 302, 303, 307, 308)) {
                    val next = conn.getHeaderField("Location")
                        ?: throw IOException("redirect $code without Location")
                    if (++redirects > 6) throw IOException("too many redirects")
                    current = URL(URL(current), next).toString()
                    continue
                }
                if (code != HttpURLConnection.HTTP_OK) throw IOException("HTTP $code")
                val digest = MessageDigest.getInstance("SHA-256")
                DigestInputStream(conn.inputStream, digest).use { input ->
                    FileOutputStream(dest).use { output ->
                        val buf = ByteArray(1 shl 16)
                        while (true) {
                            val n = input.read(buf)
                            if (n < 0) break
                            output.write(buf, 0, n)
                        }
                        output.fd.sync()
                    }
                }
                return digest.digest().joinToString("") { "%02x".format(it) }
            } finally {
                conn.disconnect()
            }
        }
    }

    // AssetInputStream.available() is the remaining uncompressed length, so the model
    // usually lands in one array; read to EOF anyway, since the API calls it an estimate.
    private fun loadAsset(assets: AssetManager, name: String): ByteArray = assets.open("speech/$name").use { input ->
        val head = ByteArray(input.available())
        var n = 0
        while (n < head.size) {
            val read = input.read(head, n, head.size - n)
            if (read < 0) break
            n += read
        }
        val rest = input.readBytes()
        if (n == head.size && rest.isEmpty()) head else head.copyOf(n) + rest
    }
}
