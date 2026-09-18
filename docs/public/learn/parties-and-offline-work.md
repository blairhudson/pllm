# Parties and offline work

Learn what each party does before and during a private inference request.

[View canonical HTML](https://pllm.run/learn/parties-and-offline-work/)

Document ID: `pllm.docs.learn.parties-and-offline-work`  
Release: `0.1.0`  
Build: `sha256:25f93731fb643fee39e706e33566ffa98bc3397f98a6a12e261bafd33b06038e`  
Source hash: `sha256:53abc51659043bf19e250a922629dbc0049f9196a8f95d35f217ea8fe996ce92`

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
