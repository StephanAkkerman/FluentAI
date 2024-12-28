import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  const body = await request.json();
  const res = await fetch("http://127.0.0.1:8765", {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
    // If you need cookies or auth info passed along:
    credentials: 'include',
  });

  const data = await res.json();
  return NextResponse.json(data);
}

