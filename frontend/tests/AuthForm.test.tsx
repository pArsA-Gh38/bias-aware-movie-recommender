import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AuthForm } from "@/components/AuthForm";

const replace = vi.fn();
const refresh = vi.fn();

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace, refresh }),
}));

describe("AuthForm", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("logs in and navigates without storing a browser token", async () => {
    vi.mocked(fetch).mockResolvedValue(Response.json({ id: "u1" }));
    const storageSpy = vi.spyOn(Storage.prototype, "setItem");
    render(<AuthForm mode="login" />);

    await userEvent.type(screen.getByLabelText("Email"), "person@example.com");
    await userEvent.type(screen.getByLabelText("Password"), "password123");
    await userEvent.click(screen.getByRole("button", { name: "Sign in" }));

    expect(fetch).toHaveBeenCalledWith("/api/auth/login", expect.objectContaining({ method: "POST" }));
    expect(storageSpy).not.toHaveBeenCalled();
    expect(replace).toHaveBeenCalledWith("/movies");
    expect(refresh).toHaveBeenCalledOnce();
  });

  it("shows backend authentication errors", async () => {
    vi.mocked(fetch).mockResolvedValue(Response.json({ detail: "Could not validate credentials" }, { status: 401 }));
    render(<AuthForm mode="login" />);
    await userEvent.type(screen.getByLabelText("Email"), "person@example.com");
    await userEvent.type(screen.getByLabelText("Password"), "wrongpass");
    await userEvent.click(screen.getByRole("button", { name: "Sign in" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not validate credentials");
  });

  it("leaves the login pending state and reports a network failure", async () => {
    vi.mocked(fetch).mockRejectedValueOnce(new TypeError("network failed"));
    render(<AuthForm mode="login" />);
    await userEvent.type(screen.getByLabelText("Email"), "person@example.com");
    await userEvent.type(screen.getByLabelText("Password"), "password123");
    await userEvent.click(screen.getByRole("button", { name: "Sign in" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Application service unavailable");
    expect(screen.getByRole("button", { name: "Sign in" })).toBeEnabled();
  });

  it("exposes an accessible pending state while login is in flight", async () => {
    let resolveRequest!: (response: Response) => void;
    vi.mocked(fetch).mockReturnValueOnce(new Promise((resolve) => { resolveRequest = resolve; }));
    render(<AuthForm mode="login" />);

    await userEvent.type(screen.getByLabelText("Email"), "person@example.com");
    await userEvent.type(screen.getByLabelText("Password"), "password123");
    await userEvent.click(screen.getByRole("button", { name: "Sign in" }));

    const pendingButton = screen.getByRole("button", { name: "Signing in…" });
    expect(pendingButton).toBeDisabled();
    expect(pendingButton).toHaveAttribute("aria-busy", "true");
    expect(screen.getByRole("status")).toHaveTextContent("Signing in, please wait.");

    resolveRequest(Response.json({ id: "u1" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Sign in" })).toBeEnabled());
  });

  it("keeps the login field and registration-link contracts accessible", () => {
    render(<AuthForm mode="login" />);

    const email = screen.getByLabelText("Email");
    expect(email).toHaveAttribute("name", "email");
    expect(email).toHaveAttribute("type", "email");
    expect(email).toHaveAttribute("autocomplete", "email");
    expect(email).toBeRequired();

    const password = screen.getByLabelText("Password");
    expect(password).toHaveAttribute("name", "password");
    expect(password).toHaveAttribute("type", "password");
    expect(password).toHaveAttribute("autocomplete", "current-password");
    expect(password).toHaveAttribute("minlength", "8");
    expect(password).toHaveAttribute("maxlength", "128");
    expect(password).toBeRequired();

    expect(screen.getByRole("link", { name: "Create an account" })).toHaveAttribute("href", "/register");
  });

  it("registers with a new password and retains successful navigation", async () => {
    vi.mocked(fetch).mockResolvedValue(Response.json({ id: "u1" }, { status: 201 }));
    render(<AuthForm mode="register" />);

    await userEvent.type(screen.getByLabelText("Email"), "person@example.com");
    const password = screen.getByLabelText("Password");
    expect(password).toHaveAttribute("autocomplete", "new-password");
    await userEvent.type(password, "password123");
    await userEvent.click(screen.getByRole("button", { name: "Register" }));

    expect(fetch).toHaveBeenCalledWith("/api/auth/register", expect.objectContaining({ method: "POST" }));
    expect(replace).toHaveBeenCalledWith("/movies");
    expect(refresh).toHaveBeenCalledOnce();
  });

  it("keeps the register presentation and field contracts accessible", () => {
    const { container } = render(<AuthForm mode="register" />);

    expect(screen.getByRole("heading", { level: 1, name: "Create an account" })).toBeVisible();
    expect(screen.getByText("Movie Recommender")).toBeVisible();

    const email = screen.getByLabelText("Email");
    expect(email).toHaveAttribute("name", "email");
    expect(email).toHaveAttribute("type", "email");
    expect(email).toHaveAttribute("autocomplete", "email");
    expect(email).toBeRequired();

    const password = screen.getByLabelText("Password");
    expect(password).toHaveAttribute("name", "password");
    expect(password).toHaveAttribute("type", "password");
    expect(password).toHaveAttribute("autocomplete", "new-password");
    expect(password).toHaveAttribute("minlength", "8");
    expect(password).toHaveAttribute("maxlength", "128");
    expect(password).toBeRequired();

    expect(container.querySelectorAll("form")).toHaveLength(1);
    const decorativeStrips = Array.from(container.querySelectorAll('[aria-hidden="true"]'))
      .filter((element) => element.children.length === 5);
    expect(decorativeStrips).toHaveLength(1);
    expect(decorativeStrips[0]).toHaveAttribute("aria-hidden", "true");
    expect(screen.getByRole("link", { name: "Sign in" })).toHaveAttribute("href", "/login");
  });

  it("registers without storing a browser token", async () => {
    vi.mocked(fetch).mockResolvedValue(Response.json({ id: "u1" }, { status: 201 }));
    const storageSpy = vi.spyOn(Storage.prototype, "setItem");
    render(<AuthForm mode="register" />);

    await userEvent.type(screen.getByLabelText("Email"), "person@example.com");
    await userEvent.type(screen.getByLabelText("Password"), "password123");
    await userEvent.click(screen.getByRole("button", { name: "Register" }));

    expect(fetch).toHaveBeenCalledWith("/api/auth/register", expect.objectContaining({ method: "POST" }));
    expect(storageSpy).not.toHaveBeenCalled();
    expect(replace).toHaveBeenCalledWith("/movies");
    expect(refresh).toHaveBeenCalledOnce();
  });

  it("exposes an accessible pending state while registration is in flight", async () => {
    let resolveRequest!: (response: Response) => void;
    vi.mocked(fetch).mockReturnValueOnce(new Promise((resolve) => { resolveRequest = resolve; }));
    render(<AuthForm mode="register" />);

    await userEvent.type(screen.getByLabelText("Email"), "person@example.com");
    await userEvent.type(screen.getByLabelText("Password"), "password123");
    await userEvent.click(screen.getByRole("button", { name: "Register" }));

    const pendingButton = screen.getByRole("button", { name: "Please wait…" });
    expect(pendingButton).toBeDisabled();
    expect(pendingButton).toHaveAttribute("aria-busy", "true");
    expect(screen.getByRole("status")).toHaveTextContent("Registering, please wait.");

    resolveRequest(Response.json({ id: "u1" }, { status: 201 }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Register" })).toBeEnabled());
  });

  it("shows backend registration errors", async () => {
    vi.mocked(fetch).mockResolvedValue(Response.json({ detail: "Email is already registered" }, { status: 409 }));
    render(<AuthForm mode="register" />);
    await userEvent.type(screen.getByLabelText("Email"), "person@example.com");
    await userEvent.type(screen.getByLabelText("Password"), "password123");
    await userEvent.click(screen.getByRole("button", { name: "Register" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Email is already registered");
  });

  it("leaves the register pending state and reports a network failure", async () => {
    vi.mocked(fetch).mockRejectedValueOnce(new TypeError("network failed"));
    render(<AuthForm mode="register" />);
    await userEvent.type(screen.getByLabelText("Email"), "person@example.com");
    await userEvent.type(screen.getByLabelText("Password"), "password123");
    await userEvent.click(screen.getByRole("button", { name: "Register" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Application service unavailable");
    expect(screen.getByRole("button", { name: "Register" })).toBeEnabled();
  });
});
