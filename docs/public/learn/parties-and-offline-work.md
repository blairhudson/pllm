# Parties and offline work

Learn what each party does before and during a private inference request.

[View canonical HTML](https://pllm.run/learn/parties-and-offline-work/)

Document ID: `pllm.docs.learn.parties-and-offline-work`  
Release: `0.1.0`

The **client** holds plaintext and authorizes each request. The **preparation
service** creates one-time protocol material before a request. The **inference
service** consumes that material while evaluating the model. A **provider**
implements a component. An **operator** controls a deployed process and its policy.

For public-weight inference, preparation commits to the model, transformer body,
stages, quantization, and retry limit. It expands seeds supplied by the client into
input masks, output masks, and one-time tickets. It computes corrections, sends
them to inference, and waits for confirmation. During inference, only the client
and inference service exchange tickets and masked activations.

Reserved rows can be used only once. Cancellation, failure, replay, or early
completion invalidates every unused row in that reservation. See
[preparation](/sdk/pipeline/preparation/) and [deployment](/sdk/operate/deployment/).
