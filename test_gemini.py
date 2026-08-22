"""
test_gemini -- everything about the Gemini adapters except the wire.

Run: python test_gemini.py

There is no outbound network on the machine this was built on, so the HTTP call
is injected. That still covers what actually breaks in an API adapter: auth
header choice, request shape, and reading a response that is not the happy path.
What it cannot cover is whether Google accepts the body, which is why the module
says UNTESTED and the model id is a parameter.

The assertion that matters beyond plumbing is `test_model_claims_are_asserted`.
A vision model is a party making claims about a place it was not present at. If
its output could ever land as `measured`, the whole support ladder in
reasoning.py becomes decoration.
"""
from __future__ import annotations

import base64
import json

import numpy as np

import gemini as gm
import provenance as P
import reasoning as rz

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))


def fake(reply, sink=None):
    """A transport that records what it was sent and returns a canned reply."""
    def transport(url, body, headers, timeout=gm.TIMEOUT):
        if sink is not None:
            sink.update(url=url, body=body, headers=headers)
        if isinstance(reply, Exception):
            raise reply
        return reply
    return transport


def text_reply(text):
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def test_auth_style():
    check("an AIza key goes in the api-key header",
          "x-goog-api-key" in gm.auth_headers("AIzaSyExample"))
    check("an AQ. token goes in as a bearer",
          gm.auth_headers("AQ.Ab8Example").get("Authorization", "").startswith("Bearer "))
    check("a ya29. token does too",
          "Authorization" in gm.auth_headers("ya29.Example"))
    check("the choice can be forced",
          "x-goog-api-key" in gm.auth_headers("AQ.x", style="key"))


def test_request_shape():
    sink = {}
    frame = np.full((40, 60, 3), 90, np.uint8)
    v = gm.GeminiVision(token="AIzaTest", transport=fake(text_reply('["a wall"]'), sink))
    v("what is off frame?", frame)

    check("hits the generateContent endpoint",
          sink["url"].endswith(":generateContent"), sink["url"].split("/")[-1])
    check("model id is in the path", "gemini" in sink["url"])
    parts = sink["body"]["contents"][0]["parts"]
    check("the prompt is sent", any("off frame" in p.get("text", "") for p in parts))
    check("the frame is sent inline",
          any("inline_data" in p for p in parts))
    check("image is valid base64 png",
          base64.b64decode(parts[1]["inline_data"]["data"])[:4] == b"\x89PNG")
    check("json is requested back",
          sink["body"]["generationConfig"]["responseMimeType"] == "application/json")
    check("a claim cap is stated in the prompt",
          "at most 4" in parts[0]["text"])


def test_response_parsing():
    v = lambda reply: gm.GeminiVision(token="AIzaTest", transport=fake(reply))

    got = v(text_reply('["fire escape, left", "wet asphalt, right"]'))("p")
    check("a json array parses", got == ["fire escape, left", "wet asphalt, right"], str(got))

    got = v(text_reply('{"claims": ["neon sign, left"]}'))("p")
    check("a wrapped object parses", got == ["neon sign, left"], str(got))

    got = v(text_reply("Probably a street."))("p")
    check("prose degrades to one claim rather than crashing",
          got == ["Probably a street."], str(got))

    check("an empty candidate list yields nothing",
          v({"candidates": []})("p") == [])
    check("a malformed payload yields nothing",
          v({"nope": 1})("p") == [])
    check("a transport failure yields nothing",
          v(OSError("connection reset"))("p") == [])

    long_reply = json.dumps(["a", "b", "c", "d", "e", "f"])
    check("the claim cap is enforced on the way back",
          len(v(text_reply(long_reply))("p")) == 4)


def test_no_credential():
    """
    "No token" stopped meaning "no credential" once ADC was wired.

    A machine with Application Default Credentials has a usable bearer token
    without anyone pasting a key, which is the point -- so the refusal has to
    key off having NOTHING, not off having no API key.
    """
    check("no credential at all means no call and no crash",
          gm.GeminiVision(token="", project="")("p") == [])
    check("no project means ADC is not even consulted",
          gm.GeminiVision(token="", project="").credential() == "")


def test_auth_route():
    """Which endpoint a credential implies."""
    print("choosing an endpoint from the credential")
    keyed = gm.GeminiVision(token="AIzaEXAMPLE", project="p-123")
    check("an explicit key goes to the public API",
          keyed.endpoint().startswith(gm.BASE), keyed.endpoint()[:48])
    check("and that key is used as-is", keyed.credential() == "AIzaEXAMPLE")

    adc = gm.GeminiVision(token="", project="p-123")
    url = adc.endpoint()
    check("keyless with a project goes to Vertex",
          gm.VERTEX_HOST in url and "/projects/p-123/" in url, url[:70])
    check("the model is named in the Vertex path",
          url.endswith(f"/{adc.model}:generateContent"))

    regional = gm.GeminiVision(token="", project="p-123", location="us-central1")
    check("a region prefixes the host, global does not",
          regional.endpoint().startswith("https://us-central1-" + gm.VERTEX_HOST))

    # a cached token is reused rather than re-minted per frame
    gm._ADC.clear()
    gm._ADC.update(token="ya29.cached", expires=2 ** 40)
    check("a live token is cached, not re-minted every call",
          gm.adc_token() == "ya29.cached")
    gm._ADC.clear()


def test_model_claims_are_asserted():
    """The line that keeps a model from laundering a guess into evidence."""
    import offscreen as off
    import test_offscreen as tof

    frames, truth = tof.make_excursion()
    h, w = frames[0].shape[:2]
    Hs = [np.eye(3) for _ in frames]
    ex = off.find_excursions(off.link_tracks(off.detect_moving(frames, Hs=Hs),
                                             frame_size=(w, h)), w, h)

    vision = gm.GeminiVision(token="AIzaTest",
                             transport=fake(text_reply('["a burning skyline, right"]')))
    r = rz.ApiReasoner(call=vision)
    brief = rz.brief_for(dict(shot=0, motion="LOCKED"), n_frames=len(frames),
                         wing_w=int(w * 0.22), frame_w=w, frame_h=h, excursions=ex)
    plan = r.plan(brief)

    model_els = [e for e in plan.elements if "skyline" in e.text]
    check("the model's claim arrives", len(model_els) == 1)
    if model_els:
        check("and it is asserted, never measured",
              model_els[0].support == "asserted", model_els[0].support)
        check("attributed to the model", "vision model" in model_els[0].because)
    check("measured elements are untouched by it",
          len(plan.by_support("measured")) == len(ex), f"{len(ex)} excursions")
    check("pixels from this plan are still outside the headline number",
          plan.label() not in P.PHOTOGRAPHIC and plan.label() in P.NOT_THIS_PLACE)


def test_helper_wires_together():
    r = gm.reasoner(token="AIzaTest")
    check("the helper returns a wired ApiReasoner", isinstance(r, rz.ApiReasoner))
    check("with a Gemini call attached", isinstance(r.call, gm.GeminiVision))


def test_image_edit_guards():
    g = gm.GeminiImageEdit(token="")
    canvas = np.zeros((20, 60, 3), np.uint8)
    hole = np.zeros((20, 60), bool)
    hole[:, :10] = True
    try:
        g(canvas, hole, None)
        check("editing without a credential refuses", False)
    except RuntimeError as e:
        check("editing without a credential refuses", "credential" in str(e).lower())

    import cv2
    img = np.full((20, 60, 3), 200, np.uint8)
    ok, buf = cv2.imencode(".png", img)
    reply = {"candidates": [{"content": {"parts": [
        {"inline_data": {"data": base64.b64encode(buf).decode()}}]}}]}
    g2 = gm.GeminiImageEdit(token="AIzaTest", transport=fake(reply))
    out = g2(canvas, hole, None)
    check("a returned image comes back the canvas shape",
          out.shape == canvas.shape, str(out.shape))

    g3 = gm.GeminiImageEdit(token="AIzaTest", transport=fake(text_reply("sorry")))
    try:
        g3(canvas, hole, None)
        check("a reply with no image is an error, not a blank wall", False)
    except RuntimeError:
        check("a reply with no image is an error, not a blank wall", True)


if __name__ == "__main__":
    print("credentials")
    test_auth_style()
    test_no_credential()
    test_auth_route()
    print("request")
    test_request_shape()
    print("response")
    test_response_parsing()
    print("support ladder")
    test_model_claims_are_asserted()
    print("wiring")
    test_helper_wires_together()
    print("image editing")
    test_image_edit_guards()

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print("  FAILED:", f)
        raise SystemExit(1)
